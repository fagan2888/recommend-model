#coding=utf8


import getopt
import string
import json
import os
import sys
import logging
sys.path.append('shell')
import click
import config
import pandas as pd
import numpy as np
import time

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund
from db.asset_stock_factor import *
from db.asset_stock import *
from db import asset_trade_dates
from multiprocessing import Pool
import math
import scipy.stats as stats
import json


logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def sf(ctx):
    '''multi factor
    '''
    pass


def call_factor_func(f):
    return f()


@sf.command()
@click.pass_context
def compute_stock_factor(ctx):



    #ln_capital_factor()
    #factor_funcs = [ln_capital_factor, ln_price_factor, highlow_price_factor, relative_strength_factor, std_factor, trade_volumn_factor, turn_rate_factor, weighted_strength_factor]
    factor_funcs = [bp_factor, current_ratio_factor, cash_ratio_factor, pe_ttm_factor, roa_factor, roe_factor, holder_factor]
    pool = Pool(10)
    pool.map(call_factor_func, factor_funcs)
    pool.close()
    pool.join()
    #highlow_price_factor()
    #bp_factor()
    #asset_turnover_factor()
    #current_ratio_factor()
    #cash_ratio_factor()
    #pe_ttm_factor()
    #roa_factor()
    #roe_factor()
    #grossprofit_factor()
    #profit_factor()
    #holder_factor()
    #fcfp_factor()

    return


@sf.command()
@click.option('--filepath', 'optfilepath', help=u'stock factor infos')
@click.pass_context
def insert_factor_info(ctx, optfilepath):
    '''insert factor info
    '''
    sf_df = pd.read_csv(optfilepath.strip())
    sf_df.factor_formula = sf_df.factor_formula.where((pd.notnull(sf_df.factor_formula)), None)

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    for i in range(0, len(sf_df)):

        record = sf_df.iloc[i]

        factor = stock_factor()
        factor.sf_id = record.sf_id
        factor.sf_name = record.factor_name
        factor.sf_explain = record.factor_explain
        factor.sf_source = record.factor_source
        factor.sf_kind = record.factor_kind
        factor.sf_formula = record.factor_formula
        factor.sf_start_date = record.start_date

        session.merge(factor)

    session.commit()
    session.close()

#所有股票代码
def all_stock_info():

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    all_stocks = pd.read_sql(session.query(ra_stock.globalid, ra_stock.sk_secode, ra_stock.sk_compcode, ra_stock.sk_name).statement, session.bind, index_col = ['sk_secode'])
    session.commit()
    session.close()

    return all_stocks


def all_stock_listdate():

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    all_stocks = pd.read_sql(session.query(ra_stock.globalid, ra_stock.sk_secode, ra_stock.sk_listdate).statement, session.bind, index_col = ['sk_secode'])
    session.commit()
    session.close()

    return all_stocks

#st股票表
def stock_st():

    all_stocks = all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_sk_specialtrade.secode, tq_sk_specialtrade.selecteddate,
            tq_sk_specialtrade.outdate).filter(tq_sk_specialtrade.selectedtype <= 3).filter(tq_sk_specialtrade.secode.in_(set(all_stocks.index))).statement
    st_stocks = pd.read_sql(sql, session.bind, index_col = ['secode'], parse_dates = ['selecteddate', 'outdate'])
    session.commit()
    session.close()

    st_stocks = pd.merge(st_stocks, all_stocks, left_index=True, right_index=True)

    return st_stocks


#取有因子值的最后一个日期
def stock_factor_last_date(sf_id, stock_id):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    result = session.query(stock_factor_value.trade_date).filter(and_(stock_factor_value.stock_id == stock_id, stock_factor_value.sf_id == sf_id)).order_by(stock_factor_value.trade_date.desc()).first()

    session.commit()
    session.close()

    if result == None:
        return datetime(1990,1,1)
    else:
        return result[0]


#股价因子
def ln_price_factor():

    all_stocks = all_stock_info()
    sf_id = 'SF.000002'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    data = {}
    for i in range(0, len(all_stocks.index)):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print 'ln price', globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(10)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.tclose).filter(and_(tq_qt_skdailyprice.secode == secode, tq_qt_skdailyprice.tradedate >= last_date)).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate']).replace(0.0, np.nan)
        if len(quotation) == 0:
            continue
        quotation = quotation.sort_index()
        data[globalid] = np.log(quotation.tclose)

    session.commit()
    session.close()

    ln_price_df = pd.DataFrame(data)

    ln_price_df = ln_price_df.loc[ln_price_df.index & month_last_day()] #取每月最后一个交易日的因子值
    ln_price_df = valid_stock_filter(ln_price_df) #过滤掉不合法的因子值
    ln_price_df = normalized(ln_price_df) #因子值归一化
    update_factor_value(sf_id, ln_price_df) #因子值存入数据库

    return


#计算财报因子
def financial_report_data(fr_df, name):

    all_stocks = all_stock_info()
    stock_listdate = all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index('sk_compcode')

    #print naps_df
    #过滤未上市时的报表信息
    groups = []
    for compcode, group in fr_df.groupby(fr_df.index):
        listdate = all_stocks.loc[compcode, 'sk_listdate']
        group = group[group.enddate > listdate]
        groups.append(group)
    fr_df = pd.concat(groups, axis = 0)


    #根据公布日期，选最近的月收盘交易日
    month_last_trade_dates = month_last_day()
    firstpublishdates = fr_df.firstpublishdate.ravel()
    trans_firstpublishdate = []
    for firstpublishdate in firstpublishdates:
        greater_month_last_trade_dates = month_last_trade_dates[month_last_trade_dates >= firstpublishdate]
        if len(greater_month_last_trade_dates) == 0:
            trans_firstpublishdate.append(np.nan)
        else:
            trans_firstpublishdate.append(greater_month_last_trade_dates[0])
    fr_df.firstpublishdate = trans_firstpublishdate
    fr_df = fr_df.dropna()

    fr_df = fr_df.reset_index()
    fr_df = fr_df.set_index(['firstpublishdate','compcode'])
    fr_df = fr_df.sort_index()
    #过滤掉两个季报一起公布的情况，比如有的公司年报和一季报一起公布
    fr_df = fr_df.groupby(level = [0, 1]).last()

    fr_df = fr_df[[name]]
    fr_df = fr_df.unstack()
    fr_df.columns = fr_df.columns.droplevel(0)

    #扩展到每个月最后一个交易日
    fr_df = fr_df.loc[month_last_day()]

    #向后填充四个月度
    fr_df = fr_df.fillna(method = 'pad', limit = 4)


    compcode_globalid = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))
    fr_df = fr_df.rename(columns = compcode_globalid)

    return fr_df


#计算季度累计数据
def quarter_aggregate(df, name):

    records = []
    for compcode, group in df.groupby(df.index):
        #print group
        group = group.sort_values('enddate', ascending = True)
        group = group.reset_index()
        group = group.set_index(['enddate'])
        for year, year_group in group.groupby(group.index.strftime('%Y')):
            #如果数据不是从年份的一季度开始，则过滤掉这一年
            if year_group.index[0].strftime('%m%d') != '0331':
                continue
            year_group[name] = year_group[name].rolling(window = 2, min_periods = 1).apply(lambda x: x[1] - x[0] if len(x) > 1 else x[0])
            records.append(year_group)

    df = pd.concat(records, axis = 0)
    df = df.reset_index()
    df = df.set_index(['compcode'])

    return df


#bp因子
def bp_factor():

    all_stocks = all_stock_info()
    stock_listdate = all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000005'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.naps).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    naps_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    naps_df = financial_report_data(naps_df, 'naps')

    sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.secode ,tq_qt_skdailyprice.tclose).filter(tq_qt_skdailyprice.tradedate.in_(naps_df.index.strftime('%Y%m%d'))).statement
    tclose_df = pd.read_sql(sql, session.bind, index_col = ['tradedate','secode'], parse_dates = ['tradedate']).replace(0.0, np.nan)
    tclose_df = tclose_df.unstack()
    tclose_df.columns = tclose_df.columns.droplevel(0)

    compcode_globalid = dict(zip(all_stocks.sk_compcode.ravel(), all_stocks.globalid.ravel()))
    secode_globalid = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    tclose_df = tclose_df.rename(columns = secode_globalid)
    columns = list(set(naps_df.columns) & set(tclose_df.columns))

    naps_df = naps_df[columns]
    tclose_df = tclose_df[columns]

    bp_df = naps_df / tclose_df

    bp_df = valid_stock_filter(bp_df)
    bp_df = normalized(bp_df)
    update_factor_value(sf_id, bp_df)

    session.commit()
    session.close()

    return

'''
#资产周转率因子
def asset_turnover_factor():

    all_stocks = all_stock_info()
    stock_listdate = all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000003'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.taturnrt).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    turnover_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    turnover_df = financial_report_data(turnover_df, 'taturnrt')

    turnover_df = valid_stock_filter(turnover_df)
    turnover_df = normalized(turnover_df)
    update_factor_value(sf_id, turnover_df)

    session.commit()
    session.close()

    return
'''


#流动比率因子
def current_ratio_factor():

    all_stocks = all_stock_info()
    stock_listdate = all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000007'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.currentrt).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    current_ratio_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    current_ratio_df = financial_report_data(current_ratio_df, 'currentrt')

    current_ratio_df = valid_stock_filter(current_ratio_df)
    current_ratio_df = normalized(current_ratio_df)
    update_factor_value(sf_id, current_ratio_df)

    session.commit()
    session.close()

    return


#现金比率因子
def cash_ratio_factor():

    all_stocks = all_stock_info()
    stock_listdate = all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000007'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.cashrt).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    cash_ratio_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    cash_ratio_df = financial_report_data(cash_ratio_df, 'cashrt')

    cash_ratio_df = valid_stock_filter(cash_ratio_df)
    cash_ratio_df = normalized(cash_ratio_df)
    update_factor_value(sf_id, cash_ratio_df)

    session.commit()
    session.close()

    return


#市盈率因子
def pe_ttm_factor():

    all_stocks = all_stock_info()
    sf_id = 'SF.000009'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    data_pe_ttm = {}
    data_pe_ttm_cut = {}
    for i in range(0, len(all_stocks.index)):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        #print 'ep', globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(10)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_sk_finindic.tradedate, tq_sk_finindic.pettm, tq_sk_finindic.pettmnpaaei).filter(and_(tq_sk_finindic.secode == secode, tq_sk_finindic.tradedate >= last_date)).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate']).replace(0.0, np.nan)
        if len(quotation) == 0:
            continue
        quotation = quotation.sort_index()
        quotation = 1.0 / quotation
        data_pe_ttm[globalid] = quotation.pettm
        data_pe_ttm_cut[globalid] = quotation.pettmnpaaei

    session.commit()
    session.close()

    ep_df = pd.DataFrame(data_pe_ttm)
    ep_cut_df = pd.DataFrame(data_pe_ttm_cut)


    ep_df = ep_df.loc[ep_df.index & month_last_day()] #取每月最后一个交易日的因子值
    ep_df = valid_stock_filter(ep_df) #过滤掉不合法的因子值
    ep_df = normalized(ep_df) #因子值归一化
    update_factor_value(sf_id, ep_df) #因子值存入数据库


    sf_id = 'SF.000010'
    ep_cut_df = ep_cut_df.loc[ep_cut_df.index & month_last_day()] #取每月最后一个交易日的因子值
    ep_cut_df = valid_stock_filter(ep_cut_df) #过滤掉不合法的因子值
    ep_cut_df = normalized(ep_cut_df) #因子值归一化
    update_factor_value(sf_id, ep_cut_df) #因子值存入数据库

    return



#roa因子
def roa_factor():

    all_stocks = all_stock_info()
    stock_listdate = all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000039'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.roa).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    roa_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    roa_df = quarter_aggregate(roa_df, 'roa')
    roa_df = financial_report_data(roa_df, 'roa')

    roa_df = valid_stock_filter(roa_df)
    roa_df = normalized(roa_df)
    update_factor_value(sf_id, roa_df)


    '''
    sf_id = 'SF.000040'
    roa_ttm_df = valid_stock_filter(roa_ttm_df)
    roa_ttm_df = normalized(roa_ttm_df)
    update_factor_value(sf_id, roa_ttm_df)
    '''

    session.commit()
    session.close()

    return



#roe因子
def roe_factor():

    all_stocks = all_stock_info()
    stock_listdate = all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000041'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.roediluted).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    roe_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    roe_df = quarter_aggregate(roe_df, 'roediluted')
    roe_df = financial_report_data(roe_df, 'roediluted')

    roe_df = valid_stock_filter(roe_df)
    roe_df = normalized(roe_df)
    update_factor_value(sf_id, roe_df)


    '''
    sf_id = 'SF.000042'
    roe_ttm_df = valid_stock_filter(roe_ttm_df)
    roe_ttm_df = normalized(roe_ttm_df)
    update_factor_value(sf_id, roe_ttm_df)
    '''

    session.commit()
    session.close()

    return



#毛利率因子
def grossprofit_factor():

    all_stocks = all_stock_info()
    stock_listdate = all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000013'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.sgpmargin).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    grossprofit_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])
    grossprofit_df = financial_report_data(grossprofit_df, 'sgpmargin')

    print grossprofit_df.iloc[:,0:15]


    '''
    sql = session.query(tq_fin_proindicdatasub.enddate ,tq_fin_proindicdatasub.firstpublishdate, tq_fin_proindicdatasub.compcode, tq_fin_proindicdatasub.reporttype ,tq_fin_proindicdatasub.grossprofit).filter(tq_fin_proindicdatasub.reporttype == 3).filter(tq_fin_proindicdatasub.compcode.in_(all_stocks.sk_compcode)).statement
    grossprofit_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])
    grossprofit_df = financial_report_data(grossprofit_df, 'grossprofit')

    print '---------------------------'
    print grossprofit_df.iloc[:,0:15]
    sys.exit(0)
    grossprofit_df = valid_stock_filter(grossprofit_df)
    grossprofit_df = normalized(grossprofit_df)
    update_factor_value(sf_id, grossprofit_df)
    '''

    grossprofit_df = valid_stock_filter(grossprofit_df)
    grossprofit_df = normalized(grossprofit_df)
    update_factor_value(sf_id, grossprofit_df)

    session.commit()
    session.close()

    return


#户均持股比例因子
def holder_factor():

    all_stocks = all_stock_info()
    stock_listdate = all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000013'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_shareholdernum.enddate ,tq_sk_shareholdernum.publishdate, tq_sk_shareholdernum.compcode, tq_sk_shareholdernum.askshamt ,tq_sk_shareholdernum.aholdproportionpacc ,tq_sk_shareholdernum.aproportiongrq, tq_sk_shareholdernum.aproportiongrhalfyear).filter(tq_sk_shareholdernum.compcode.in_(all_stocks.sk_compcode)).statement
    holder_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['publishdate', 'enddate'])

    holder_df = holder_df.rename(columns={'publishdate':'firstpublishdate'})

    aholdproportionpacc_holder_df = holder_df[['enddate', 'firstpublishdate', 'aholdproportionpacc']]
    aproportiongrq_holder_df = holder_df[['enddate', 'firstpublishdate', 'aproportiongrq']]
    aproportiongrhalfyear_holder_df = holder_df[['enddate', 'firstpublishdate', 'aproportiongrhalfyear']]
    #print holder_df.head()

    aholdproportionpacc_holder_df = financial_report_data(aholdproportionpacc_holder_df, 'aholdproportionpacc')
    aproportiongrq_holder_df = financial_report_data(aproportiongrq_holder_df, 'aproportiongrq')
    aproportiongrhalfyear_holder_df = financial_report_data(aproportiongrhalfyear_holder_df, 'aproportiongrhalfyear')

    sf_id = 'SF.000019'
    aholdproportionpacc_holder_df = valid_stock_filter(aholdproportionpacc_holder_df)
    aholdproportionpacc_holder_df = normalized(aholdproportionpacc_holder_df)
    update_factor_value(sf_id, aholdproportionpacc_holder_df)

    sf_id = 'SF.000021'
    aproportiongrq_holder_df = valid_stock_filter(aproportiongrq_holder_df)
    aproportiongrq_holder_df = normalized(aproportiongrq_holder_df)
    update_factor_value(sf_id, aproportiongrq_holder_df)

    sf_id = 'SF.000020'
    aproportiongrhalfyear_holder_df = valid_stock_filter(aproportiongrhalfyear_holder_df)
    aproportiongrhalfyear_holder_df = normalized(aproportiongrhalfyear_holder_df)
    update_factor_value(sf_id, aproportiongrhalfyear_holder_df)

    session.commit()
    session.close()

    return


def fcfp_factor():

    all_stocks = all_stock_info()
    stock_listdate = all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000013'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.fcff).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    fcff_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    fcff_df = fcff_df.iloc[0:1000,]
    fcff_df = financial_report_data(fcff_df, 'fcff')
    print fcff_df.iloc[:,0:15]

    session.commit()
    session.close()

    return


'''
#扣非净利润
def profit_factor():

    all_stocks = all_stock_info()
    stock_listdate = all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000013'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.npcut).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    profit_growth_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    #print profit_growth_df

    for compcode, group in profit_growth_df.groupby(profit_growth_df.index):
        #print group
        group = group.sort_values('enddate', ascending = True)


    profit_growth_df = financial_report_data(profit_growth_df, 'sgpmargin')

    profit_df = valid_stock_filter(profit_growth_df)
    profit_df = normalized(profit_growth_df)
    update_factor_value(sf_id, profit_growth_df)

    session.commit()
    session.close()

    return
'''



@sf.command()
@click.pass_context
#股票合法性表
def insert_stock_valid(ctx):

    all_stocks = all_stock_info()
    st_stocks = stock_st()
    list_date = all_stock_listdate()
    #print list_date.head()
    list_date.sk_listdate = list_date.sk_listdate + timedelta(365)
    #print list_date.head()

    #print all_stocks
    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.secode ,tq_qt_skdailyprice.tclose).filter(tq_qt_skdailyprice.secode.in_(all_stocks.index)).statement
    #print sql

    #过滤停牌股票
    quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']).replace(0.0, np.nan)
    quotation = quotation.unstack()
    quotation.columns = quotation.columns.droplevel(0)
    session.commit()
    session.close()

    #60个交易日内需要有25个交易日未停牌
    quotation_count = quotation.rolling(60).count()
    quotation[quotation_count < 25] = np.nan


    #过滤st股票
    for i in range(0, len(st_stocks)):
        secode = st_stocks.index[i]
        record = st_stocks.iloc[i]
        selecteddate = record.selecteddate
        outdate = record.outdate
        if secode in set(quotation.columns):
            #print secode, selecteddate, outdate
            quotation.loc[selecteddate:outdate, secode] = np.nan

    #过滤上市未满一年股票
    for secode in list_date.index:
        if secode in set(quotation.columns):
            #print secode, list_date.loc[secode, 'sk_listdate']
            quotation.loc[:list_date.loc[secode, 'sk_listdate'], secode] = np.nan


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()
    for date in quotation.index:
        print date
        records = []
        for secode in quotation.columns:
            globalid = all_stocks.loc[secode, 'globalid']
            value = quotation.loc[date, secode]
            if np.isnan(value):
                continue
            valid = 1.0
            stock_valid = stock_factor_stock_valid()
            stock_valid.stock_id = globalid
            stock_valid.secode = secode
            stock_valid.trade_date = date
            stock_valid.valid = valid

            records.append(stock_valid)
            #session.merge(stock_valid)

        session.add_all(records)
        session.commit()

    session.close()
    #print np.sum(np.sum(pd.isnull(quotation)))
    #print np.sum(np.sum(~pd.isnull(quotation)))

    pass


#过滤掉不合法股票
def valid_stock_filter(factor_df):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    for stock_id in factor_df.columns:
        print 'valid stock id : ', stock_id
        sql = session.query(stock_factor_stock_valid.trade_date, stock_factor_stock_valid.valid).filter(stock_factor_stock_valid.stock_id == stock_id).statement
        valid_df = pd.read_sql(sql, session.bind, index_col = ['trade_date'], parse_dates = ['trade_date'])
        valid_df = valid_df[valid_df.valid == 1]
        if len(factor_df) == 0:
            facto_df.stock_id = np.nan
        else:
            factor_df[stock_id][~factor_df.index.isin(valid_df.index)] = np.nan

    #print factor_df[factor_df != np.nan]
    #print record

    session.commit()
    session.close()

    return factor_df


#归一化
def normalized(factor_df):

    factor_median = factor_df.median(axis = 1)
    factor_std  = factor_df.std(axis = 1)
    factor_mean  = factor_df.mean(axis = 1)
    factor_std = factor_std.dropna()#一行中只有一个数据，则没有标准差，std后为nan

    factor_median = factor_median.loc[factor_std.index]
    factor_df = factor_df.loc[factor_std.index]

    factor_df = factor_df.sub(factor_median, axis = 0)
    factor_df = factor_df.div(factor_std, axis = 0)

    factor_df[factor_df > 5]  = 5
    factor_df[factor_df < -5] = -5

    return factor_df



#获取月收盘日
def month_last_day():

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()

    trade_dates = asset_trade_dates.trade_dates
    sql = session.query(trade_dates.td_date).filter(trade_dates.td_type >= 8).statement
    trade_df = pd.read_sql(sql, session.bind, index_col = ['td_date'], parse_dates = ['td_date'])
    trade_df = trade_df.sort_index()
    session.commit()
    session.close()

    return trade_df.index


#更新因子值数据
def update_factor_value(sf_id, factor_df):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    for stock_id in factor_df.columns:
        ser = factor_df[stock_id].dropna()
        print 'update : ' , sf_id ,stock_id
        #records = []
        for date in ser.index:
            sfv = stock_factor_value()
            sfv.sf_id = sf_id
            sfv.stock_id = stock_id
            sfv.trade_date = date
            sfv.factor_value = ser.loc[date]
            #records.append(sfv)
            session.merge(sfv)

        #session.add_all(records)
        session.commit()

    session.close()

    return


#高低股价因子
def highlow_price_factor():

    all_stocks = all_stock_info()
    sf_id = 'SF.000016'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_12m = {}
    data_6m = {}
    data_3m = {}
    data_1m = {}

    for i in range(0, len(all_stocks.index)):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print 'highlow price', globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(370)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_sk_dquoteindic.tradedate, tq_sk_dquoteindic.tcloseaf).filter(tq_sk_dquoteindic.secode == secode).filter(tq_sk_dquoteindic.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
        if len(quotation) == 0:
            continue
        quotation = quotation.sort_index()
        trade_dates = quotation.index
        quotation = quotation.reindex(pd.date_range(quotation.index.min(), quotation.index.max()))
        tclose_12m_max = quotation.tcloseaf.rolling(360, 1).max().loc[trade_dates[360:]]
        tclose_6m_max = quotation.tcloseaf.rolling(180, 1).max().loc[trade_dates[180:]]
        tclose_3m_max = quotation.tcloseaf.rolling(90, 1).max().loc[trade_dates[90:]]
        tclose_1m_max = quotation.tcloseaf.rolling(30, 1).max().loc[trade_dates[30:]]
        tclose_12m_min = quotation.tcloseaf.rolling(360, 1).min().loc[trade_dates[360:]]
        tclose_6m_min = quotation.tcloseaf.rolling(180, 1).min().loc[trade_dates[180:]]
        tclose_3m_min = quotation.tcloseaf.rolling(90, 1).min().loc[trade_dates[90:]]
        tclose_1m_min = quotation.tcloseaf.rolling(30, 1).min().loc[trade_dates[30:]]
        data_12m[globalid] = tclose_12m_max / tclose_12m_min
        data_6m[globalid] = tclose_6m_max / tclose_6m_min
        data_3m[globalid] = tclose_3m_max / tclose_3m_min
        data_1m[globalid] = tclose_1m_max / tclose_1m_min

    session.commit()
    session.close()

    highlow_price_12m_df = pd.DataFrame(data_12m)
    highlow_price_6m_df = pd.DataFrame(data_6m)
    highlow_price_3m_df = pd.DataFrame(data_3m)
    highlow_price_1m_df = pd.DataFrame(data_1m)

    highlow_price_12m_df = highlow_price_12m_df.loc[highlow_price_12m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    highlow_price_6m_df = highlow_price_6m_df.loc[highlow_price_6m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    highlow_price_3m_df = highlow_price_3m_df.loc[highlow_price_3m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    highlow_price_1m_df = highlow_price_1m_df.loc[highlow_price_1m_df.index & month_last_day()] #取每月最后一个交易日的因子值

    highlow_price_12m_df = valid_stock_filter(highlow_price_12m_df)
    highlow_price_6m_df = valid_stock_filter(highlow_price_6m_df)
    highlow_price_3m_df = valid_stock_filter(highlow_price_3m_df)
    highlow_price_1m_df = valid_stock_filter(highlow_price_1m_df)

    highlow_price_12m_df = normalized(highlow_price_12m_df)
    highlow_price_6m_df = normalized(highlow_price_6m_df)
    highlow_price_3m_df = normalized(highlow_price_3m_df)
    highlow_price_1m_df = normalized(highlow_price_1m_df)

    update_factor_value('SF.000015', highlow_price_12m_df)
    update_factor_value('SF.000018', highlow_price_6m_df)
    update_factor_value('SF.000017', highlow_price_3m_df)
    update_factor_value('SF.000016', highlow_price_1m_df)

    return


#动量因子
def relative_strength_factor():

    all_stocks = all_stock_info()
    sf_id = 'SF.000036'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_6m = {}
    data_3m = {}
    data_1m = {}
    data_12m= {}

    for i in range(0, len(all_stocks.index)):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print 'relative strength', globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(10)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.yieldy, tq_sk_yieldindic.yieldm, tq_sk_yieldindic.yield3m, tq_sk_yieldindic.yield6m).filter(tq_sk_yieldindic.secode == secode).filter(tq_sk_yieldindic.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
        if len(quotation) == 0:
            continue
        quotation = quotation.sort_index()
        quotation = quotation / 100.0

        data_6m[globalid] = quotation.yield6m
        data_3m[globalid] = quotation.yield3m
        data_1m[globalid] = quotation.yieldm
        data_12m[globalid] = quotation.yieldy

    session.commit()
    session.close()

    relative_strength_6m_df = pd.DataFrame(data_6m)
    relative_strength_3m_df = pd.DataFrame(data_3m)
    relative_strength_1m_df = pd.DataFrame(data_1m)
    relative_strength_12m_df = pd.DataFrame(data_12m)

    relative_strength_12m_df = relative_strength_12m_df.loc[relative_strength_12m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    relative_strength_6m_df = relative_strength_6m_df.loc[relative_strength_6m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    relative_strength_3m_df = relative_strength_3m_df.loc[relative_strength_3m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    relative_strength_1m_df = relative_strength_1m_df.loc[relative_strength_1m_df.index & month_last_day()] #取每月最后一个交易日的因子值


    relative_strength_12m_df = valid_stock_filter(relative_strength_12m_df)
    relative_strength_6m_df = valid_stock_filter(relative_strength_6m_df)
    relative_strength_3m_df = valid_stock_filter(relative_strength_3m_df)
    relative_strength_1m_df = valid_stock_filter(relative_strength_1m_df)

    relative_strength_12m_df = normalized(relative_strength_12m_df)
    relative_strength_6m_df = normalized(relative_strength_6m_df)
    relative_strength_3m_df = normalized(relative_strength_3m_df)
    relative_strength_1m_df = normalized(relative_strength_1m_df)

    update_factor_value('SF.000035', relative_strength_12m_df)
    update_factor_value('SF.000038', relative_strength_6m_df)
    update_factor_value('SF.000037', relative_strength_3m_df)
    update_factor_value('SF.000036', relative_strength_1m_df)

    return



#波动率因子
def std_factor():

    all_stocks = all_stock_info()
    sf_id = 'SF.000048'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_6m = {}
    data_3m = {}
    data_1m = {}
    data_12m= {}

    for i in range(0, len(all_stocks.index)):

        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print 'std', globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(370)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode == secode).filter(tq_sk_yieldindic.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
        if len(quotation) == 0:
            continue
        quotation = quotation.sort_index()
        trade_dates = quotation.index
        quotation = quotation / 100.0
        quotation = quotation.reindex(pd.date_range(quotation.index.min(), quotation.index.max()))

        yield_12m_std = quotation.Yield.rolling(360, 1).std().loc[trade_dates[360:]]
        yield_6m_std = quotation.Yield.rolling(180, 1).std().loc[trade_dates[180:]]
        yield_3m_std = quotation.Yield.rolling(90, 1).std().loc[trade_dates[90:]]
        yield_1m_std = quotation.Yield.rolling(30, 1).std().loc[trade_dates[30:]]

        data_12m[globalid] = yield_12m_std
        data_6m[globalid] = yield_6m_std
        data_3m[globalid] = yield_3m_std
        data_1m[globalid] = yield_1m_std

    session.commit()
    session.close()

    std_12m_df = pd.DataFrame(data_12m)
    std_6m_df = pd.DataFrame(data_6m)
    std_3m_df = pd.DataFrame(data_3m)
    std_1m_df = pd.DataFrame(data_1m)

    std_12m_df = std_12m_df.loc[std_12m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    std_6m_df = std_6m_df.loc[std_6m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    std_3m_df = std_3m_df.loc[std_3m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    std_1m_df = std_1m_df.loc[std_1m_df.index & month_last_day()] #取每月最后一个交易日的因子值

    std_12m_df = valid_stock_filter(std_12m_df)
    std_6m_df = valid_stock_filter(std_6m_df)
    std_3m_df = valid_stock_filter(std_3m_df)
    std_1m_df = valid_stock_filter(std_1m_df)

    std_12m_df = normalized(std_12m_df)
    std_6m_df = normalized(std_6m_df)
    std_3m_df = normalized(std_3m_df)
    std_1m_df = normalized(std_1m_df)

    update_factor_value('SF.000047', std_12m_df)
    update_factor_value('SF.000050', std_6m_df)
    update_factor_value('SF.000049', std_3m_df)
    update_factor_value('SF.000048', std_1m_df)


    return


#成交金额因子
def trade_volumn_factor():

    all_stocks = all_stock_info()
    sf_id = 'SF.000052'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_6m = {}
    data_3m = {}
    data_1m = {}
    data_12m= {}

    for i in range(0, len(all_stocks.index)):

        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print 'trade volumn' , globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(370)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.amount).filter(tq_qt_skdailyprice.secode == secode).filter(tq_qt_skdailyprice.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
        if len(quotation) == 0:
            continue
        quotation = quotation.sort_index()
        trade_dates = quotation.index
        quotation = quotation / 10000.0
        quotation = quotation.reindex(pd.date_range(quotation.index.min(), quotation.index.max()))
        #print quotation

        amount_12m_sum = quotation.amount.rolling(360, 1).sum().loc[trade_dates[360:]]
        amount_6m_sum = quotation.amount.rolling(180, 1).sum().loc[trade_dates[180:]]
        amount_3m_sum = quotation.amount.rolling(90, 1).sum().loc[trade_dates[90:]]
        amount_1m_sum = quotation.amount.rolling(30, 1).sum().loc[trade_dates[30:]]

        data_12m[globalid] = amount_12m_sum
        data_6m[globalid] = amount_6m_sum
        data_3m[globalid] = amount_3m_sum
        data_1m[globalid] = amount_1m_sum

    session.commit()
    session.close()

    amount_12m_df = pd.DataFrame(data_12m)
    amount_6m_df = pd.DataFrame(data_6m)
    amount_3m_df = pd.DataFrame(data_3m)
    amount_1m_df = pd.DataFrame(data_1m)

    amount_12m_df = amount_12m_df.loc[amount_12m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    amount_6m_df = amount_6m_df.loc[amount_6m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    amount_3m_df = amount_3m_df.loc[amount_3m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    amount_1m_df = amount_1m_df.loc[amount_1m_df.index & month_last_day()] #取每月最后一个交易日的因子值

    amount_12m_df = valid_stock_filter(amount_12m_df)
    amount_6m_df = valid_stock_filter(amount_6m_df)
    amount_3m_df = valid_stock_filter(amount_3m_df)
    amount_1m_df = valid_stock_filter(amount_1m_df)

    amount_12m_df = normalized(amount_12m_df)
    amount_6m_df = normalized(amount_6m_df)
    amount_3m_df = normalized(amount_3m_df)
    amount_1m_df = normalized(amount_1m_df)

    update_factor_value('SF.000051', amount_12m_df)
    update_factor_value('SF.000054', amount_6m_df)
    update_factor_value('SF.000053', amount_3m_df)
    update_factor_value('SF.000052', amount_1m_df)


    return


#换手率因子
def turn_rate_factor():

    all_stocks = all_stock_info()
    sf_id = 'SF.000056'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_6m = {}
    data_3m = {}
    data_1m = {}
    data_12m= {}

    for i in range(0, len(all_stocks.index)):

        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print 'turn rate', globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(10)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.turnrate, tq_sk_yieldindic.turnratem, tq_sk_yieldindic.turnrate3m,
                tq_sk_yieldindic.turnrate6m, tq_sk_yieldindic.turnratey).filter(tq_sk_yieldindic.secode == secode).filter(tq_sk_yieldindic.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
        if len(quotation) == 0:
            continue
        quotation = quotation.sort_index()
        quotation = quotation / 100.0
        quotation = quotation.reindex(pd.date_range(quotation.index.min(), quotation.index.max()))
        #print quotation

        data_12m[globalid] = quotation.turnratey
        data_6m[globalid] = quotation.turnrate6m
        data_3m[globalid] = quotation.turnrate3m
        data_1m[globalid] = quotation.turnratem

    session.commit()
    session.close()

    turnrate_12m_df = pd.DataFrame(data_12m)
    turnrate_6m_df = pd.DataFrame(data_6m)
    turnrate_3m_df = pd.DataFrame(data_3m)
    turnrate_1m_df = pd.DataFrame(data_1m)

    turnrate_12m_df = turnrate_12m_df.loc[turnrate_12m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    turnrate_6m_df = turnrate_6m_df.loc[turnrate_6m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    turnrate_3m_df = turnrate_3m_df.loc[turnrate_3m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    turnrate_1m_df = turnrate_1m_df.loc[turnrate_1m_df.index & month_last_day()] #取每月最后一个交易日的因子值

    turnrate_12m_df = valid_stock_filter(turnrate_12m_df)
    turnrate_6m_df = valid_stock_filter(turnrate_6m_df)
    turnrate_3m_df = valid_stock_filter(turnrate_3m_df)
    turnrate_1m_df = valid_stock_filter(turnrate_1m_df)

    turnrate_12m_df = normalized(turnrate_12m_df)
    turnrate_6m_df = normalized(turnrate_6m_df)
    turnrate_3m_df = normalized(turnrate_3m_df)
    turnrate_1m_df = normalized(turnrate_1m_df)

    update_factor_value('SF.000055', turnrate_12m_df)
    update_factor_value('SF.000058', turnrate_6m_df)
    update_factor_value('SF.000057', turnrate_3m_df)
    update_factor_value('SF.000056', turnrate_1m_df)

    return


#加权动量因子
def weighted_strength_factor():

    all_stocks = all_stock_info()
    sf_id = 'SF.000060'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_6m = {}
    data_3m = {}
    data_1m = {}
    data_12m= {}

    for i in range(0, len(all_stocks.index)):

        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print 'weight strength' , globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(370)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.turnrate, tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode == secode).filter(tq_sk_yieldindic.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
        if len(quotation) == 0:
            continue
        quotation = quotation.sort_index()
        trade_dates = quotation.index
        quotation = quotation / 100.0
        quotation = quotation.reindex(pd.date_range(quotation.index.min(), quotation.index.max()))
        quotation['weighted_strength'] = quotation.turnrate * quotation.Yield
        #print quotation

        data_12m[globalid] = quotation.weighted_strength.rolling(360, 1).mean().loc[trade_dates[360:]]
        data_6m[globalid] = quotation.weighted_strength.rolling(180, 1).mean().loc[trade_dates[180:]]
        data_3m[globalid] = quotation.weighted_strength.rolling(90, 1).mean().loc[trade_dates[90:]]
        data_1m[globalid] = quotation.weighted_strength.rolling(30, 1).mean().loc[trade_dates[30:]]

    session.commit()
    session.close()

    weighted_strength_12m_df = pd.DataFrame(data_12m)
    weighted_strength_6m_df = pd.DataFrame(data_6m)
    weighted_strength_3m_df = pd.DataFrame(data_3m)
    weighted_strength_1m_df = pd.DataFrame(data_1m)

    weighted_strength_12m_df = weighted_strength_12m_df.loc[weighted_strength_12m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    weighted_strength_6m_df = weighted_strength_6m_df.loc[weighted_strength_6m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    weighted_strength_3m_df = weighted_strength_3m_df.loc[weighted_strength_3m_df.index & month_last_day()] #取每月最后一个交易日的因子值
    weighted_strength_1m_df = weighted_strength_1m_df.loc[weighted_strength_1m_df.index & month_last_day()] #取每月最后一个交易日的因子值

    weighted_strength_12m_df = valid_stock_filter(weighted_strength_12m_df)
    weighted_strength_6m_df = valid_stock_filter(weighted_strength_6m_df)
    weighted_strength_3m_df = valid_stock_filter(weighted_strength_3m_df)
    weighted_strength_1m_df = valid_stock_filter(weighted_strength_1m_df)

    weighted_strength_12m_df = normalized(weighted_strength_12m_df)
    weighted_strength_6m_df = normalized(weighted_strength_6m_df)
    weighted_strength_3m_df = normalized(weighted_strength_3m_df)
    weighted_strength_1m_df = normalized(weighted_strength_1m_df)

    update_factor_value('SF.000059', weighted_strength_12m_df)
    update_factor_value('SF.000062', weighted_strength_6m_df)
    update_factor_value('SF.000061', weighted_strength_3m_df)
    update_factor_value('SF.000060', weighted_strength_1m_df)

    return


#市值因子
def ln_capital_factor():

    all_stocks = all_stock_info()
    sf_id = 'SF.000001'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data = {}

    for i in range(0, len(all_stocks.index)):

        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print 'capital ' , globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(10)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.totmktcap).filter(tq_qt_skdailyprice.secode == secode).filter(tq_qt_skdailyprice.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate']).replace(0.0, np.nan)
        if len(quotation) == 0:
            continue
        quotation = quotation.sort_index()
        data[globalid] = np.log(quotation.totmktcap)

    ln_capital_df = pd.DataFrame(data)

    session.commit()
    session.close()

    ln_capital_df = ln_capital_df.loc[ln_capital_df.index & month_last_day()] #取每月最后一个交易日的因子值

    ln_capital_df = valid_stock_filter(ln_capital_df)

    ln_capital_df = normalized(ln_capital_df)

    update_factor_value('SF.000001', ln_capital_df)

    return



@sf.command()
@click.pass_context
def compute_stock_factor_layer_rankcorr(ctx):

    sf_ids = ['SF.000001','SF.000002', 'SF.000015', 'SF.000016','SF.000017','SF.000018', 'SF.000035', 'SF.000036', 'SF.000037', 'SF.000038', 
            'SF.000047','SF.000048','SF.000049','SF.000050','SF.000051','SF.000052','SF.000053','SF.000054','SF.000055','SF.000056',
            'SF.000057','SF.000058','SF.000059','SF.000060','SF.000061','SF.000062','SF.000005','SF.000007','SF.000009','SF.000019','SF.000020',
            'SF.000021','SF.000039','SF.000041',
            ]


    pool = Pool(20)
    pool.map(stock_factor_layer_spearman, sf_ids)
    pool.close()
    pool.join()

    #stock_factor_layer_spearman('SF.000041')

    return



def stock_factor_layer_spearman(sf_id):

    all_stocks = all_stock_info()
    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index('globalid')

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    #sf_id = 'SF.000001'

    sql = session.query(stock_factor_value.stock_id, stock_factor_value.trade_date, stock_factor_value.factor_value).filter(stock_factor_value.sf_id == sf_id).statement
    factor_value_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'stock_id'], parse_dates = ['trade_date'])
    factor_value_df = factor_value_df.unstack()

    session.commit()
    session.close()


    layer_dates = []
    layers = []

    for date in factor_value_df.index:
        ser = factor_value_df.loc[date]
        ser = ser.dropna()
        ser.index = ser.index.droplevel(0)

        #按照因子值大小分为5档
        ser = ser.sort_values(ascending = False)
        ser_df = pd.DataFrame(ser)
        ser_df = pd.concat([ser_df, all_stocks], axis = 1, join_axes = [ser_df.index])

        ser_df['order'] = range(0, len(ser_df))
        ser_df.order = ser_df.order / (math.ceil(len(ser_df) / 5.0))
        ser_df.order = ser_df.order.astype(int)

        ser_df = ser_df.reset_index()
        ser_df = ser_df.set_index(['sk_secode'])

        #print ser_df
        layers.append(ser_df)
        layer_dates.append(date)

    rankcorrdates = []
    rankcorrs = []

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    for i in range(0, len(layer_dates) - 1):
        layer_date = layer_dates[i]
        yield_date = layer_dates[i + 1]
        ser_df = layers[i]

        #print date
        sql = session.query(tq_sk_yieldindic.secode, tq_sk_yieldindic.yieldm).filter(tq_sk_yieldindic.tradedate == yield_date.strftime('%Y%m%d')).statement
        yieldm_df = pd.read_sql(sql, session.bind, index_col = ['secode']) / 100
        yieldm_df = pd.concat([ser_df, yieldm_df], axis = 1, join_axes = [ser_df.index])
        yieldm_df = yieldm_df[['order', 'yieldm']]
        #print date, 1.0 * np.sum(pd.isnull(yieldm_df)['yieldm'])/ len(yieldm_df)
        yieldm_df = yieldm_df.dropna()#财会数据库数据有稍许缺失，但缺失量都在1%以下，2010年以后基本没有数据缺失情况，可以忽略

        yieldm_df = yieldm_df.reset_index()
        yieldm_df = yieldm_df.set_index(['order'])
        yieldm_df = yieldm_df.drop(['sk_secode'])
        order_yieldm_df = yieldm_df.groupby(yieldm_df.index).mean()
        order_yieldm = order_yieldm_df.yieldm
        #print order_yieldm.ravel()
        #print order_yieldm.index.ravel()
        spearmanr = -1.0 * stats.stats.spearmanr(order_yieldm, order_yieldm.index)[0]
        print sf_id, layer_date, spearmanr

        rankcorrs.append(spearmanr)
        rankcorrdates.append(layer_date)

    session.commit()
    session.close()


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    for i in range(0, len(rankcorrdates)):

        factor_rankcorr = stock_factor_rankcorr()
        factor_rankcorr.sf_id = sf_id
        factor_rankcorr.trade_date = rankcorrdates[i]
        factor_rankcorr.rankcorr = rankcorrs[i]

        session.merge(factor_rankcorr)


    for i in range(0, len(layer_dates)):

        layer_df = layers[i]

        for order, group in layer_df.groupby(layer_df.order):
            json_stock_ids = json.dumps(list(group.stock_id.ravel()))

            factor_layer = stock_factor_layer()
            factor_layer.sf_id = sf_id
            factor_layer.trade_date = layer_dates[i]
            factor_layer.layer = order
            factor_layer.stock_ids = json_stock_ids
            session.merge(factor_layer)

    session.commit()
    session.close()

    return



@sf.command()
@click.pass_context
def compute_stock_factor_index(ctx):

    sf_ids = ['SF.000001','SF.000002', 'SF.000015', 'SF.000016','SF.000017','SF.000018', 'SF.000035', 'SF.000036', 'SF.000037', 'SF.000038', 
            'SF.000047','SF.000048','SF.000049','SF.000050','SF.000051','SF.000052','SF.000053','SF.000054','SF.000055','SF.000056',
            'SF.000057','SF.000058','SF.000059','SF.000060','SF.000061','SF.000062','SF.000005','SF.000007','SF.000009','SF.000019','SF.000020',
            'SF.000021','SF.000039','SF.000041',
            ]


    pool = Pool(16)
    pool.map(stock_factor_index, sf_ids)
    pool.close()
    pool.join()

    #stock_factor_index('SF.000041')
    return


def stock_factor_index(sf_id):

    all_stocks = all_stock_info()
    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index('globalid')
    month_last_trade_dates = month_last_day().tolist()

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(stock_factor_rankcorr.sf_id, stock_factor_rankcorr.trade_date, stock_factor_rankcorr.rankcorr).statement
    rankcorr_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'sf_id'], parse_dates = ['trade_date'])
    rankcorr_df = rankcorr_df.unstack()
    rankcorr_df.columns = rankcorr_df.columns.droplevel(0)

    rankcorr_df = rankcorr_df.rolling(14).mean()
    sf_rankcorr = rankcorr_df[sf_id]
    #print sf_rankcorr

    #获取每期选因子的哪端
    stock_factor_pos = {}
    sf_rankcorr = rankcorr_df[sf_id]
    for date in sf_rankcorr.index.tolist():
        rankcorr = sf_rankcorr.loc[date]
        if rankcorr >= 0:
            layer = 0
        else:
            layer = 4
        date_index = month_last_trade_dates.index(date)
        stock_date = month_last_trade_dates[date_index + 1]
        record = session.query(stock_factor_layer.stock_ids).filter(and_(stock_factor_layer.sf_id == sf_id, stock_factor_layer.layer == layer, stock_factor_layer.trade_date == stock_date)).first()
        if record is None:
            continue
        stock_factor_pos[stock_date] = json.loads(record[0])

    globalid_secode_dict = dict(zip(all_stocks.index.ravel(), all_stocks.sk_secode.ravel()))

    #计算每期股票的仓位
    dates = list(stock_factor_pos.keys())
    dates.sort()
    stock_factor_pos_df = pd.DataFrame(0, index = dates, columns = all_stocks.index)
    for date in dates:
        stocks = stock_factor_pos[date]
        record = stock_factor_pos_df.loc[date]
        record[record.index.isin(stocks)] = 1.0 / len(stocks)
        stock_factor_pos_df.loc[date] = record
    stock_factor_pos_df = stock_factor_pos_df.rename(columns = globalid_secode_dict)



    #计算因子指数
    caihui_engine = database.connection('caihui')
    caihui_Session = sessionmaker(bind=caihui_engine)
    caihui_session = caihui_Session()

    sql = caihui_session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.secode ,tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.tradedate >= stock_factor_pos_df.index[0].strftime('%Y%m%d')).statement
    stock_yield_df = pd.read_sql(sql, caihui_session.bind, index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']) / 100.0
    stock_yield_df = stock_yield_df.unstack()
    stock_yield_df.columns = stock_yield_df.columns.droplevel(0)
    secodes = list(set(stock_factor_pos_df.columns) & set(stock_yield_df.columns))
    stock_factor_pos_df = stock_factor_pos_df[secodes]
    stock_yield_df = stock_yield_df[secodes]
    caihui_session.commit()
    caihui_session.close()


    stock_factor_pos_df = stock_factor_pos_df.reindex(stock_yield_df.index).fillna(method = 'pad')
    stock_factor_pos_df = stock_factor_pos_df.shift(1).fillna(0.0)

    factor_yield_df = stock_factor_pos_df * stock_yield_df
    factor_yield_df = factor_yield_df.sum(axis = 1)
    factor_nav_df = (1 + factor_yield_df).cumprod()

    print sf_id, factor_nav_df.index[-1], factor_nav_df.iloc[-1]

    for date in factor_nav_df.index:

        sfn = stock_factor_nav()
        sfn.sf_id = sf_id
        sfn.trade_date = date
        sfn.nav = factor_nav_df.loc[date]
        session.merge(sfn)

    session.commit()
    session.close()

    return


@sf.command()
@click.pass_context
def select_stock_factor_layer(ctx):

    all_stocks = all_stock_info()
    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index('globalid')

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    factor_type = session.query(stock_factor.sf_id, stock_factor.sf_kind).all()
    factor_type_dict = {}
    for record in factor_type:
        factor_type_dict[record[0]] = record[1]
    #print factor_type_dict

    sql = session.query(stock_factor_rankcorr.sf_id, stock_factor_rankcorr.trade_date, stock_factor_rankcorr.rankcorr).statement
    rankcorr_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'sf_id'], parse_dates = ['trade_date'])

    rankcorr_df = rankcorr_df.unstack()
    rankcorr_df.columns = rankcorr_df.columns.droplevel(0)

    rankcorr_df = rankcorr_df.rolling(14).mean().iloc[14:,]
    rankcorr_abs_df = abs(rankcorr_df)

    stock_pos = {}
    factor_pos = {}
    for date in rankcorr_abs_df.index:
        rankcorr_abs = rankcorr_abs_df.loc[date]
        rankcorr_abs = rankcorr_abs.sort_values(ascending = False)

        date_stocks = stock_pos.setdefault(date, [])
        date_factors = factor_pos.setdefault(date, [])

        has_type = set()
        for index in rankcorr_abs.index:
            rankcorr = rankcorr_df.loc[date, index]
            layer = None
            if rankcorr >= 0:
                layer = 0
            else:
                layer = 4

            record = session.query(stock_factor_layer.stock_ids).filter(and_(stock_factor_layer.sf_id == index,stock_factor_layer.trade_date == date,
                                    stock_factor_layer.layer == layer)).first()

            if record is None:
                continue

            factor_type = factor_type_dict[index]
            if factor_type in has_type:
                continue
            else:
                has_type.add(factor_type)

            date_stocks.extend(json.loads(record[0]))
            date_factors.append([index, layer])

            if len(date_factors) >= 5:
                break

    factor_pos_df = pd.DataFrame(factor_pos).T
    print factor_pos_df

    #计算每期股票的仓位
    globalid_secode_dict = dict(zip(all_stocks.index.ravel(), all_stocks.sk_secode.ravel()))
    dates = list(stock_pos.keys())
    dates.sort()
    stock_pos_df = pd.DataFrame(0, index = dates, columns = all_stocks.index)
    for date in dates:
        stocks = stock_pos[date]
        stocks = list(set(stocks))
        record = stock_pos_df.loc[date]
        record[record.index.isin(stocks)] = 1.0 / len(stocks)
        stock_pos_df.loc[date] = record
    stock_pos_df = stock_pos_df.rename(columns = globalid_secode_dict)

    session.commit()
    session.close()


    caihui_engine = database.connection('caihui')
    caihui_Session = sessionmaker(bind=caihui_engine)
    caihui_session = caihui_Session()

    sql = caihui_session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.secode ,tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.tradedate >= stock_pos_df.index[0].strftime('%Y%m%d')).statement
    stock_yield_df = pd.read_sql(sql, caihui_session.bind, index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']) / 100.0
    stock_yield_df = stock_yield_df.unstack()
    stock_yield_df.columns = stock_yield_df.columns.droplevel(0)
    secodes = list(set(stock_pos_df.columns) & set(stock_yield_df.columns))
    stock_pos_df = stock_pos_df[secodes]
    stock_yield_df = stock_yield_df[secodes]

    caihui_session.commit()
    caihui_session.close()

    stock_pos_df = stock_pos_df.reindex(stock_yield_df.index).fillna(method = 'pad')
    stock_pos_df = stock_pos_df.shift(1).fillna(0.0)

    factor_yield_df = stock_pos_df * stock_yield_df
    factor_yield_df = factor_yield_df.sum(axis = 1)
    factor_yield_df = factor_yield_df[factor_yield_df.index >= '2006-06-01']
    #print factor_yield_df.index
    factor_nav_df = (1 + factor_yield_df).cumprod()

    factor_nav_df = factor_nav_df.to_frame()
    factor_nav_df.index.name = 'date'
    factor_nav_df.columns = ['nav']
    #print factor_nav_df
    factor_nav_df.to_csv('factor_nav.csv')
    #print factor_nav_df.index[-1], factor_nav_df.iloc[-1]

    return
