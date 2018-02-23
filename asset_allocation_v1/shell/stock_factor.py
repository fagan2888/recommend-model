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
import Portfolio as PF
import math
import scipy.stats as stats
import json

import stock_util
import stock_factor_util

logger = logging.getLogger(__name__)


#股价因子
def ln_price_factor():

    all_stocks = stock_util.all_stock_info()
    sf_id = 'SF.000002'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    data = {}

    for i in range(0, len(all_stocks.index)):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        logger.info('ln price : ' +  str(globalid) + '\t' + str(secode))
        last_date = stock_factor_util.stock_factor_last_date(sf_id, globalid) - timedelta(10)
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

    ln_price_df = ln_price_df.loc[ln_price_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    ln_price_df = stock_factor_util.valid_stock_filter(ln_price_df) #过滤掉不合法的因子值
    ln_price_df = stock_factor_util.normalized(ln_price_df) #因子值归一化
    stock_factor_util.update_factor_value(sf_id, ln_price_df) #因子值存入数据库

    return


#bp因子
def bp_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000005'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.naps).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    naps_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    print naps_df
    sys.exit(0)

    naps_df = stock_factor_util.financial_report_data(naps_df, 'naps')

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

    bp_df = stock_factor_util.valid_stock_filter(bp_df)
    bp_df = stock_factor_util.normalized(bp_df)
    stock_factor_util.update_factor_value(sf_id, bp_df)

    session.commit()
    session.close()

    return


###资产周转率因子
##def asset_turnover_factor():
##
##    all_stocks = stock_util.all_stock_info()
##    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
##    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])
##
##    sf_id = 'SF.000003'
##
##    engine = database.connection('caihui')
##    Session = sessionmaker(bind=engine)
##    session = Session()
##
##    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.taturnrt).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
##    turnover_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])
##
##    turnover_df = stock_factor_util.financial_report_data(turnover_df, 'taturnrt')
##
##    turnover_df = stock_factor_util.valid_stock_filter(turnover_df)
##    turnover_df = stock_factor_util.normalized(turnover_df)
##    stock_factor_util.update_factor_value(sf_id, turnover_df)
##
##    session.commit()
##    session.close()
##
##    return


#流动比率因子
def current_ratio_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000007'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.currentrt).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    current_ratio_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    current_ratio_df = stock_factor_util.financial_report_data(current_ratio_df, 'currentrt')

    current_ratio_df = stock_factor_util.valid_stock_filter(current_ratio_df)
    current_ratio_df = stock_factor_util.normalized(current_ratio_df)
    stock_factor_util.update_factor_value(sf_id, current_ratio_df)

    session.commit()
    session.close()

    return


#现金比率因子
def cash_ratio_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000006'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.cashrt).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    cash_ratio_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    cash_ratio_df = stock_factor_util.financial_report_data(cash_ratio_df, 'cashrt')

    cash_ratio_df = stock_factor_util.valid_stock_filter(cash_ratio_df)
    cash_ratio_df = stock_factor_util.normalized(cash_ratio_df)
    stock_factor_util.update_factor_value(sf_id, cash_ratio_df)

    session.commit()
    session.close()

    return


#市盈率因子
def pe_ttm_factor():

    all_stocks = stock_util.all_stock_info()
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
        last_date = stock_factor_util.stock_factor_last_date(sf_id, globalid) - timedelta(10)
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


    ep_df = ep_df.loc[ep_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    ep_df = stock_factor_util.valid_stock_filter(ep_df) #过滤掉不合法的因子值
    ep_df = stock_factor_util.normalized(ep_df) #因子值归一化
    stock_factor_util.update_factor_value(sf_id, ep_df) #因子值存入数据库


    sf_id = 'SF.000010'
    ep_cut_df = ep_cut_df.loc[ep_cut_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    ep_cut_df = stock_factor_util.valid_stock_filter(ep_cut_df) #过滤掉不合法的因子值
    ep_cut_df = stock_factor_util.normalized(ep_cut_df) #因子值归一化
    stock_factor_util.update_factor_value(sf_id, ep_cut_df) #因子值存入数据库

    return



#roa因子
def roa_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000039'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.roa).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    roa_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    roa_df = stock_factor_util.quarter_aggregate(roa_df, 'roa')
    roa_df = stock_factor_util.financial_report_data(roa_df, 'roa')

    roa_df = stock_factor_util.valid_stock_filter(roa_df)
    roa_df = stock_factor_util.normalized(roa_df)
    stock_factor_util.update_factor_value(sf_id, roa_df)


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

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000041'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.roediluted).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    roe_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    roe_df = stock_factor_util.quarter_aggregate(roe_df, 'roediluted')
    roe_df = stock_factor_util.financial_report_data(roe_df, 'roediluted')

    roe_df = stock_factor_util.valid_stock_filter(roe_df)
    roe_df = stock_factor_util.normalized(roe_df)
    stock_factor_util.update_factor_value(sf_id, roe_df)


    '''
    sf_id = 'SF.000042'
    roe_ttm_df = valid_stock_filter(roe_ttm_df)
    roe_ttm_df = normalized(roe_ttm_df)
    update_factor_value(sf_id, roe_ttm_df)
    '''

    session.commit()
    session.close()

    return



#负债比率因子
def debtequityratio_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000008'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.ltmliabtoequconms).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    debtequity_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    debtequity_df = stock_factor_util.financial_report_data(debtequity_df, 'ltmliabtoequconms')

    debtequity_df = stock_factor_util.valid_stock_filter(debtequity_df)
    debtequity_df = stock_factor_util.normalized(debtequity_df)
    stock_factor_util.update_factor_value(sf_id, debtequity_df)

    session.commit()
    session.close()

    return


#金融杠杆率因子
def finalcial_leverage_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000012'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.equtotliab).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    finalcial_leverage_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    finalcial_leverage_df = 1.0 / (1.0 + finalcial_leverage_df)


    finalcial_leverage_df = stock_factor_util.financial_report_data(finalcial_leverage_df, 'equtotliab')

    finalcial_leverage_df = stock_factor_util.valid_stock_filter(finalcial_leverage_df)
    finalcial_leverage_df = stock_factor_util.normalized(finalcial_leverage_df)
    stock_factor_util.update_factor_value(sf_id, finalcial_leverage_df)

    session.commit()
    session.close()

    return


#市值杠杆率
def market_leverage_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000022'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.equtotliab).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    market_leverage_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    market_leverage_df = stock_factor_util.financial_report_data(debtequity_df, 'equtotliab')

    market_leverage_df = stock_factor_util.valid_stock_filter(market_leverage_df)
    market_leverage_df = stock_factor_util.normalized(market_leverage_df)
    stock_factor_util.update_factor_value(sf_id, market_leverage_df)

    session.commit()
    session.close()

    return






#毛利率因子
def grossprofit_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000013'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.sgpmargin).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    grossprofit_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])
    grossprofit_df = stock_factor_util.financial_report_data(grossprofit_df, 'sgpmargin')

    #print grossprofit_df.iloc[:,0:15]

    '''
    sql = session.query(tq_fin_proindicdatasub.enddate ,tq_fin_proindicdatasub.firstpublishdate, tq_fin_proindicdatasub.compcode, tq_fin_proindicdatasub.reporttype ,tq_fin_proindicdatasub.grossprofit).filter(tq_fin_proindicdatasub.reporttype == 3).filter(tq_fin_proindicdatasub.compcode.in_(all_stocks.sk_compcode)).statement
    grossprofit_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])
    grossprofit_df = stock_factor_util.financial_report_data(grossprofit_df, 'grossprofit')

    print '---------------------------'
    print grossprofit_df.iloc[:,0:15]
    sys.exit(0)
    grossprofit_df = valid_stock_filter(grossprofit_df)
    grossprofit_df = normalized(grossprofit_df)
    update_factor_value(sf_id, grossprofit_df)
    '''

    grossprofit_df = stock_factor_util.valid_stock_filter(grossprofit_df)
    grossprofit_df = stock_factor_util.normalized(grossprofit_df)
    stock_factor_util.update_factor_value(sf_id, grossprofit_df)

    session.commit()
    session.close()

    return


#户均持股比例因子
def holder_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
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

    aholdproportionpacc_holder_df = stock_factor_util.financial_report_data(aholdproportionpacc_holder_df, 'aholdproportionpacc')
    aproportiongrq_holder_df = stock_factor_util.financial_report_data(aproportiongrq_holder_df, 'aproportiongrq')
    aproportiongrhalfyear_holder_df = stock_factor_util.financial_report_data(aproportiongrhalfyear_holder_df, 'aproportiongrhalfyear')

    sf_id = 'SF.000019'
    aholdproportionpacc_holder_df = stock_factor_util.valid_stock_filter(aholdproportionpacc_holder_df)
    aholdproportionpacc_holder_df = stock_factor_util.normalized(aholdproportionpacc_holder_df)
    stock_factor_util.update_factor_value(sf_id, aholdproportionpacc_holder_df)

    sf_id = 'SF.000021'
    aproportiongrq_holder_df = stock_factor_util.valid_stock_filter(aproportiongrq_holder_df)
    aproportiongrq_holder_df = stock_factor_util.normalized(aproportiongrq_holder_df)
    stock_factor_util.update_factor_value(sf_id, aproportiongrq_holder_df)

    sf_id = 'SF.000020'
    aproportiongrhalfyear_holder_df = stock_factor_util.valid_stock_filter(aproportiongrhalfyear_holder_df)
    aproportiongrhalfyear_holder_df = stock_factor_util.normalized(aproportiongrhalfyear_holder_df)
    stock_factor_util.update_factor_value(sf_id, aproportiongrhalfyear_holder_df)

    session.commit()
    session.close()

    return



def fcfp_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    sf_id = 'SF.000013'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.fcfe).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    fcfe_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    fcfe_df = stock_factor_util.financial_report_data(fcfe_df, 'fcfe')

    sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.secode ,tq_qt_skdailyprice.totmktcap).filter(tq_qt_skdailyprice.tradedate.in_(fcfe_df.index.strftime('%Y%m%d'))).statement
    totmktcap_df = pd.read_sql(sql, session.bind, index_col = ['tradedate','secode'], parse_dates = ['tradedate']).replace(0.0, np.nan)
    totmktcap_df = totmktcap_df.unstack()
    totmktcap_df.columns = totmktcap_df.columns.droplevel(0)

    compcode_globalid = dict(zip(all_stocks.sk_compcode.ravel(), all_stocks.globalid.ravel()))
    secode_globalid = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    totmktcap_df = totmktcap_df.rename(columns = secode_globalid)
    columns = list(set(fcfe_df.columns) & set(totmktcap_df.columns))

    fcfe_df = fcfe_df[columns]
    totmktcap_df = totmktcap_df[columns]

    fcfp_df = fcfe_df / totmktcap_df

    fcfp_df = stock_factor_util.valid_stock_filter(fcfp_df)
    fcfp_df = stock_factor_util.normalized(fcfp_df)
    stock_factor_util.update_factor_value(sf_id, fcfp_df)

    session.commit()
    session.close()

    return



##扣非净利润
#def profit_factor():
#
#    all_stocks = stock_util.all_stock_info()
#    stock_listdate = all_stock_listdate()[['sk_listdate']]
#    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])
#
#    sf_id = 'SF.000013'
#
#    engine = database.connection('caihui')
#    Session = sessionmaker(bind=engine)
#    session = Session()
#
#    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.npcut).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
#    profit_growth_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])
#
#    #print profit_growth_df
#
#    for compcode, group in profit_growth_df.groupby(profit_growth_df.index):
#        #print group
#        group = group.sort_values('enddate', ascending = True)
#
#
#    profit_growth_df = financial_report_data(profit_growth_df, 'sgpmargin')
#
#    profit_df = stock_factor_util.valid_stock_filter(profit_growth_df)
#    profit_df = stock_factor_util.normalized(profit_growth_df)
#    stock_factor_util.update_factor_value(sf_id, profit_growth_df)
#
#    session.commit()
#    session.close()
#
#    return



#高低股价因子
def highlow_price_factor():

    all_stocks = stock_util.all_stock_info()
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
        logger.info('highlow price : ' +  str(globalid) + '\t' + str(secode))
        last_date = stock_factor_util.stock_factor_last_date(sf_id, globalid) - timedelta(370)
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

    highlow_price_12m_df = highlow_price_12m_df.loc[highlow_price_12m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    highlow_price_6m_df = highlow_price_6m_df.loc[highlow_price_6m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    highlow_price_3m_df = highlow_price_3m_df.loc[highlow_price_3m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    highlow_price_1m_df = highlow_price_1m_df.loc[highlow_price_1m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值

    highlow_price_12m_df = stock_factor_util.valid_stock_filter(highlow_price_12m_df)
    highlow_price_6m_df = stock_factor_util.valid_stock_filter(highlow_price_6m_df)
    highlow_price_3m_df = stock_factor_util.valid_stock_filter(highlow_price_3m_df)
    highlow_price_1m_df = stock_factor_util.valid_stock_filter(highlow_price_1m_df)

    highlow_price_12m_df = stock_factor_util.normalized(highlow_price_12m_df)
    highlow_price_6m_df = stock_factor_util.normalized(highlow_price_6m_df)
    highlow_price_3m_df = stock_factor_util.normalized(highlow_price_3m_df)
    highlow_price_1m_df = stock_factor_util.normalized(highlow_price_1m_df)

    stock_factor_util.update_factor_value('SF.000015', highlow_price_12m_df)
    stock_factor_util.update_factor_value('SF.000018', highlow_price_6m_df)
    stock_factor_util.update_factor_value('SF.000017', highlow_price_3m_df)
    stock_factor_util.update_factor_value('SF.000016', highlow_price_1m_df)

    return



#动量因子
def relative_strength_factor():

    all_stocks = stock_util.all_stock_info()
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
        logger.info('relative strength : ' +  str(globalid) + '\t' + str(secode))
        last_date = stock_factor_util.stock_factor_last_date(sf_id, globalid) - timedelta(10)
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

    relative_strength_12m_df = relative_strength_12m_df.loc[relative_strength_12m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    relative_strength_6m_df = relative_strength_6m_df.loc[relative_strength_6m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    relative_strength_3m_df = relative_strength_3m_df.loc[relative_strength_3m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    relative_strength_1m_df = relative_strength_1m_df.loc[relative_strength_1m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值


    relative_strength_12m_df = stock_factor_util.valid_stock_filter(relative_strength_12m_df)
    relative_strength_6m_df = stock_factor_util.valid_stock_filter(relative_strength_6m_df)
    relative_strength_3m_df = stock_factor_util.valid_stock_filter(relative_strength_3m_df)
    relative_strength_1m_df = stock_factor_util.valid_stock_filter(relative_strength_1m_df)

    relative_strength_12m_df = stock_factor_util.normalized(relative_strength_12m_df)
    relative_strength_6m_df = stock_factor_util.normalized(relative_strength_6m_df)
    relative_strength_3m_df = stock_factor_util.normalized(relative_strength_3m_df)
    relative_strength_1m_df = stock_factor_util.normalized(relative_strength_1m_df)

    stock_factor_util.update_factor_value('SF.000035', relative_strength_12m_df)
    stock_factor_util.update_factor_value('SF.000038', relative_strength_6m_df)
    stock_factor_util.update_factor_value('SF.000037', relative_strength_3m_df)
    stock_factor_util.update_factor_value('SF.000036', relative_strength_1m_df)

    return



#波动率因子
def std_factor():

    all_stocks = stock_util.all_stock_info()
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
        logger.info('std : ' +  str(globalid) + '\t' + str(secode))
        last_date = stock_factor_util.stock_factor_last_date(sf_id, globalid) - timedelta(370)
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

    std_12m_df = std_12m_df.loc[std_12m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    std_6m_df = std_6m_df.loc[std_6m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    std_3m_df = std_3m_df.loc[std_3m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    std_1m_df = std_1m_df.loc[std_1m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值

    std_12m_df = stock_factor_util.valid_stock_filter(std_12m_df)
    std_6m_df = stock_factor_util.valid_stock_filter(std_6m_df)
    std_3m_df = stock_factor_util.valid_stock_filter(std_3m_df)
    std_1m_df = stock_factor_util.valid_stock_filter(std_1m_df)

    std_12m_df = stock_factor_util.normalized(std_12m_df)
    std_6m_df = stock_factor_util.normalized(std_6m_df)
    std_3m_df = stock_factor_util.normalized(std_3m_df)
    std_1m_df = stock_factor_util.normalized(std_1m_df)

    stock_factor_util.update_factor_value('SF.000047', std_12m_df)
    stock_factor_util.update_factor_value('SF.000050', std_6m_df)
    stock_factor_util.update_factor_value('SF.000049', std_3m_df)
    stock_factor_util.update_factor_value('SF.000048', std_1m_df)


    return


#成交金额因子
def trade_volumn_factor():

    all_stocks = stock_util.all_stock_info()
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
        logger.info('trade volumn : ' +  str(globalid) + '\t' + str(secode))
        last_date = stock_factor_util.stock_factor_last_date(sf_id, globalid) - timedelta(370)
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

    amount_12m_df = amount_12m_df.loc[amount_12m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    amount_6m_df = amount_6m_df.loc[amount_6m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    amount_3m_df = amount_3m_df.loc[amount_3m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    amount_1m_df = amount_1m_df.loc[amount_1m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值

    amount_12m_df = stock_factor_util.valid_stock_filter(amount_12m_df)
    amount_6m_df = stock_factor_util.valid_stock_filter(amount_6m_df)
    amount_3m_df = stock_factor_util.valid_stock_filter(amount_3m_df)
    amount_1m_df = stock_factor_util.valid_stock_filter(amount_1m_df)

    amount_12m_df = stock_factor_util.normalized(amount_12m_df)
    amount_6m_df = stock_factor_util.normalized(amount_6m_df)
    amount_3m_df = stock_factor_util.normalized(amount_3m_df)
    amount_1m_df = stock_factor_util.normalized(amount_1m_df)

    stock_factor_util.update_factor_value('SF.000051', amount_12m_df)
    stock_factor_util.update_factor_value('SF.000054', amount_6m_df)
    stock_factor_util.update_factor_value('SF.000053', amount_3m_df)
    stock_factor_util.update_factor_value('SF.000052', amount_1m_df)


    return


#换手率因子
def turn_rate_factor():

    all_stocks = stock_util.all_stock_info()
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
        logger.info('turn rate : ' +  str(globalid) + '\t' + str(secode))
        last_date = stock_factor_util.stock_factor_last_date(sf_id, globalid) - timedelta(10)
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

    turnrate_12m_df = turnrate_12m_df.loc[turnrate_12m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    turnrate_6m_df = turnrate_6m_df.loc[turnrate_6m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    turnrate_3m_df = turnrate_3m_df.loc[turnrate_3m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    turnrate_1m_df = turnrate_1m_df.loc[turnrate_1m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值

    turnrate_12m_df = stock_factor_util.valid_stock_filter(turnrate_12m_df)
    turnrate_6m_df = stock_factor_util.valid_stock_filter(turnrate_6m_df)
    turnrate_3m_df = stock_factor_util.valid_stock_filter(turnrate_3m_df)
    turnrate_1m_df = stock_factor_util.valid_stock_filter(turnrate_1m_df)

    turnrate_12m_df = stock_factor_util.normalized(turnrate_12m_df)
    turnrate_6m_df = stock_factor_util.normalized(turnrate_6m_df)
    turnrate_3m_df = stock_factor_util.normalized(turnrate_3m_df)
    turnrate_1m_df = stock_factor_util.normalized(turnrate_1m_df)

    stock_factor_util.update_factor_value('SF.000055', turnrate_12m_df)
    stock_factor_util.update_factor_value('SF.000058', turnrate_6m_df)
    stock_factor_util.update_factor_value('SF.000057', turnrate_3m_df)
    stock_factor_util.update_factor_value('SF.000056', turnrate_1m_df)

    return



#加权动量因子
def weighted_strength_factor():

    all_stocks = stock_util.all_stock_info()
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
        logger.info('weight strength : ' +  str(globalid) + '\t' + str(secode))
        last_date = stock_factor_util.stock_factor_last_date(sf_id, globalid) - timedelta(370)
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

    weighted_strength_12m_df = weighted_strength_12m_df.loc[weighted_strength_12m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    weighted_strength_6m_df = weighted_strength_6m_df.loc[weighted_strength_6m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    weighted_strength_3m_df = weighted_strength_3m_df.loc[weighted_strength_3m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值
    weighted_strength_1m_df = weighted_strength_1m_df.loc[weighted_strength_1m_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值

    weighted_strength_12m_df = stock_factor_util.valid_stock_filter(weighted_strength_12m_df)
    weighted_strength_6m_df = stock_factor_util.valid_stock_filter(weighted_strength_6m_df)
    weighted_strength_3m_df = stock_factor_util.valid_stock_filter(weighted_strength_3m_df)
    weighted_strength_1m_df = stock_factor_util.valid_stock_filter(weighted_strength_1m_df)

    weighted_strength_12m_df = stock_factor_util.normalized(weighted_strength_12m_df)
    weighted_strength_6m_df = stock_factor_util.normalized(weighted_strength_6m_df)
    weighted_strength_3m_df = stock_factor_util.normalized(weighted_strength_3m_df)
    weighted_strength_1m_df = stock_factor_util.normalized(weighted_strength_1m_df)

    stock_factor_util.update_factor_value('SF.000059', weighted_strength_12m_df)
    stock_factor_util.update_factor_value('SF.000062', weighted_strength_6m_df)
    stock_factor_util.update_factor_value('SF.000061', weighted_strength_3m_df)
    stock_factor_util.update_factor_value('SF.000060', weighted_strength_1m_df)

    return


#市值因子
def ln_capital_factor():

    all_stocks = stock_util.all_stock_info()
    sf_id = 'SF.000001'

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data = {}

    for i in range(0, len(all_stocks.index)):

        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        logger.info('capital : ' +  str(globalid) + '\t' + str(secode))
        last_date = stock_factor_util.stock_factor_last_date(sf_id, globalid) - timedelta(10)
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

    ln_capital_df = ln_capital_df.loc[ln_capital_df.index & stock_factor_util.month_last_day()] #取每月最后一个交易日的因子值

    ln_capital_df = stock_factor_util.valid_stock_filter(ln_capital_df)

    ln_capital_df = stock_factor_util.normalized(ln_capital_df)

    stock_factor_util.update_factor_value('SF.000001', ln_capital_df)

    return




def compute_stock_factor_layer_spearman(layer_num, sf_id):

    all_stocks = stock_util.all_stock_info()
    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index('globalid')
    layer_num = 1.0 * layer_num

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
        ser_df.order = ser_df.order / (math.ceil(len(ser_df) / layer_num))
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
        yieldm_df = yieldm_df[['yieldm']]
        order_yieldm_df = yieldm_df.groupby(yieldm_df.index).mean()
        order_yieldm = order_yieldm_df.yieldm
        #print order_yieldm.ravel()
        #print order_yieldm.index.ravel()
        spearmanr = -1.0 * stats.stats.spearmanr(order_yieldm, order_yieldm.index)[0]
        logger.info( str(sf_id) + '\t' + str(layer_date) + '\t' + str(spearmanr) )

        rankcorrs.append(spearmanr)
        rankcorrdates.append(yield_date)

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



#计算股票因子指数
def compute_stock_factor_index(sf_id):

    all_stocks = stock_util.all_stock_info()
    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index('globalid')
    month_last_trade_dates = stock_factor_util.month_last_day().tolist()

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
        layer = 0 if rankcorr >= 0 else 9
        record = session.query(stock_factor_layer.stock_ids).filter(and_(stock_factor_layer.sf_id == sf_id, stock_factor_layer.layer == layer, stock_factor_layer.trade_date == date)).first()
        if record is None:
            continue
        stock_factor_pos[date] = json.loads(record[0])

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

    #print sf_id, factor_nav_df.index[-1], factor_nav_df.iloc[-1]

    for date in factor_nav_df.index:

        sfn = stock_factor_nav()
        sfn.sf_id = sf_id
        sfn.trade_date = date
        sfn.nav = factor_nav_df.loc[date]
        session.merge(sfn)

    session.commit()
    session.close()

    return


#计算股票差值因子指数
def compute_stock_factor_minus_index(sf_id):

    all_stocks = stock_util.all_stock_info()
    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index('globalid')
    month_last_trade_dates = stock_factor_util.month_last_day().tolist()

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()


    #获取每期选因子的两端
    sql = session.query(stock_factor_rankcorr.sf_id, stock_factor_rankcorr.trade_date, stock_factor_rankcorr.rankcorr).statement
    rankcorr_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'sf_id'], parse_dates = ['trade_date'])
    rankcorr_df = rankcorr_df.unstack()
    rankcorr_df.columns = rankcorr_df.columns.droplevel(0)


    stock_factor_positive_pos = {}
    stock_factor_negative_pos = {}
    sf_rankcorr = rankcorr_df[sf_id]
    for date in sf_rankcorr.index.tolist():
        rankcorr = sf_rankcorr.loc[date]
        positive_record = session.query(stock_factor_layer.stock_ids).filter(and_(stock_factor_layer.sf_id == sf_id, stock_factor_layer.layer == 0, stock_factor_layer.trade_date == date)).first()
        negative_record = session.query(stock_factor_layer.stock_ids).filter(and_(stock_factor_layer.sf_id == sf_id, stock_factor_layer.layer == 9, stock_factor_layer.trade_date == date)).first()
        if positive_record is None or negative_record is None:
            continue
        stock_factor_positive_pos[date] = json.loads(positive_record[0])
        stock_factor_negative_pos[date] = json.loads(negative_record[0])

    globalid_secode_dict = dict(zip(all_stocks.index.ravel(), all_stocks.sk_secode.ravel()))


    #计算每期股票的仓位
    dates = list(stock_factor_positive_pos.keys())
    dates.sort()
    stock_factor_pos_df = pd.DataFrame(0, index = dates, columns = all_stocks.index)
    for date in dates:
        positive_stocks = stock_factor_positive_pos[date]
        negative_stocks = stock_factor_negative_pos[date]
        record = stock_factor_pos_df.loc[date]
        record[record.index.isin(positive_stocks)] = 1.0 / len(record.index)
        record[record.index.isin(negative_stocks)] = -1.0 / len(record.index)
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
    factor_yield_df = factor_yield_df[factor_yield_df.index >= '2000-01-01']
    factor_nav_df = (1 + factor_yield_df).cumprod()
    factor_nav_df = factor_nav_df.to_frame()
    factor_nav_df.columns = [sf_id]

    print factor_nav_df.tail()

    session.commit()
    session.close()

    return factor_nav_df



def compute_rankcorr_multi_factor():

    all_stocks = stock_util.all_stock_info()
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
            layer = 0 if rankcorr >= 0 else 9

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
    #print factor_pos_df


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


def compute_rankcorr_multi_factor_pos(rank_num = None):

    all_stocks = stock_util.all_stock_info()
    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index('globalid')

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    factor_type = session.query(stock_factor.sf_id, stock_factor.sf_kind).all()
    factor_type_dict = {}
    for record in factor_type:
        group = factor_type_dict.setdefault(record[1], [])
        group.append(record[0])

    #print factor_type_dict
    factor_groups = []
    for key, values in factor_type_dict.items():
        factor_groups.append(values)

    #print factor_type_dict

    sql = session.query(stock_factor_rankcorr.sf_id, stock_factor_rankcorr.trade_date, stock_factor_rankcorr.rankcorr).statement
    rankcorr_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'sf_id'], parse_dates = ['trade_date'])

    rankcorr_df = rankcorr_df.unstack()
    rankcorr_df.columns = rankcorr_df.columns.droplevel(0)
    rankcorr_df = rankcorr_df.rolling(14).mean().iloc[14:,]
    rankcorr_abs_df = abs(rankcorr_df)

    factor_layer_nav_df = pd.read_csv('factor_layer_nav.csv', index_col = ['tradedate'], parse_dates = ['tradedate'])

    stock_pos = {}
    factor_pos = {}

    for date in rankcorr_abs_df.index:
        rankcorr_abs = rankcorr_abs_df.loc[date]
        factor_rankcorr = {}
        for group in factor_groups:
            group = list(set(group) & set(rankcorr_abs.index))
            kind_factors = rankcorr_abs.loc[group]
            kind_factors = kind_factors.sort_values(ascending = False)
            kind_factors = kind_factors.dropna()
            if len(kind_factors) == 1:
                factor_rankcorr[kind_factors.index[0]] = kind_factors.ravel()[0]
            elif len(kind_factors) > 1:
                factor_rankcorr[kind_factors.index[0]] = kind_factors.ravel()[0]
                factor_rankcorr[kind_factors.index[1]] = kind_factors.ravel()[1]

        factor_rankcorr = sorted(factor_rankcorr.items(), key = lambda x:x[1], reverse = True)


        date_stocks = stock_pos.setdefault(date, [])
        date_factors = factor_pos.setdefault(date, [])

        for k, v in factor_rankcorr[0:5]:
            rankcorr = rankcorr_df.loc[date, k]
            layer = 0 if rankcorr >= 0 else 9
            #print date, k, layer

            record = session.query(stock_factor_layer.stock_ids).filter(and_(stock_factor_layer.sf_id == k,
                stock_factor_layer.trade_date == date,stock_factor_layer.layer == layer)).first()

            date_factors.append([k, layer])

            if record is None:
                date_stocks.append(np.nan)
            else:
                date_stocks.extend(json.loads(record[0]))


    #print factor_pos

    #rankcorr_abs = rankcorr_abs_df.loc[date]
    #rankcorr_abs = rankcorr_abs.sort_values(ascending = False)

    #date_stocks = stock_pos.setdefault(date, [])
    #date_factors = factor_pos.setdefault(date, [])

    #start_num = 0 if rank_num is None else rank_num

    #has_type = set()
    #for i in range(start_num, len(rankcorr_abs.index)):
    #    index = rankcorr_abs.index[i]
    #    rankcorr = rankcorr_df.loc[date, index]
    #    layer = 0 if rankcorr >= 0 else 9
    #    record = session.query(stock_factor_layer.stock_ids).filter(and_(stock_factor_layer.sf_id == index,stock_factor_layer.trade_date == date,
    #                            stock_factor_layer.layer == layer)).first()

    #    if record is None:
    #        continue

    #    factor_type = factor_type_dict[index]
    #    #if factor_type in has_type:
    #    #    continue
    #    #else:
    #    #    has_type.add(factor_type)

    #    date_stocks.extend(json.loads(record[0]))
    #    date_factors.append([index, layer])

    #    if rank_num is not None:
    #        break
    #    if len(date_factors) >= 5:
    #        break


    factor_nav_df = pd.read_csv('factor_layer_nav.csv', index_col = ['tradedate'], parse_dates = ['tradedate'])

    factor_pos_df = pd.DataFrame(factor_pos).T
    factor_names = []
    for date in factor_pos_df.index:
        names = []
        for sf_id in factor_pos_df.loc[date]:
            sf_id, layer = sf_id[0], sf_id[1]
            record = session.query(stock_factor.sf_name).filter(stock_factor.sf_id == sf_id).first()
            print date, sf_id, record[0], layer
            names.append([record[0], layer])
        factor_names.append(names)

    factor_name_df = pd.DataFrame(factor_names, index = factor_pos_df.index)
    factor_name_df.to_csv('factor_name.csv')
    print factor_name_df

    dates = factor_pos_df.index[20:]
    stock_pos_df = pd.DataFrame(0, index = dates, columns = all_stocks.index)
    for date in dates:
        factor_index_data = {}
        factor_stocks = {}
        for sf_id in factor_pos_df.loc[date]:
            sf_id, layer = sf_id[0], sf_id[1]
            sf_id_layer = sf_id + '_' + str(layer)
            nav = factor_nav_df[sf_id_layer]
            nav = nav[nav.index <= date]
            nav = nav.iloc[-120:,]
            #print date,nav

            record = session.query(stock_factor_layer.stock_ids).filter(and_(stock_factor_layer.sf_id == sf_id,stock_factor_layer.trade_date == date,
                                stock_factor_layer.layer == layer)).first()

            if record is None:
                continue

            factor_index_data[sf_id] = nav
            factor_stocks[sf_id] = json.loads(record[0])

        factor_index_df = pd.DataFrame(factor_index_data)
        df_inc = factor_index_df.pct_change().fillna(0.0)

        bound = []
        for asset in df_inc.columns:
            bound.append({'sum1': 0,    'sum2' : 0,   'upper': 1.0,  'lower': 0.0})

        risk, returns, ws, sharpe = PF.markowitz_bootstrape(df_inc, bound, cpu_count=32, bootstrap_count=0)

        print date , df_inc.columns, ws

        record = pd.Series(0, index = all_stocks.index)
        for i in range(0, len(ws)):
            tmp_record = pd.Series(0, index = all_stocks.index)
            sf_id = df_inc.columns[i]
            stocks = factor_stocks[sf_id]
            w = ws[i]
            tmp_record[tmp_record.index.isin(stocks)] = w / len(stocks)
            record = record + tmp_record

        stock_pos_df.loc[date] = record


    #计算每期股票的仓位
    '''
    dates = list(stock_pos.keys())
    dates.sort()
    stock_pos_df = pd.DataFrame(0, index = dates, columns = all_stocks.index)
    for date in dates:
        stocks = stock_pos[date]
        stocks = list(stocks)
        stocks_num = 1.0 * len(stocks)
        for st_id in stocks:
           stock_pos_df.loc[date, st_id] = stocks.count(st_id) / stocks_num
    '''

    session.commit()
    session.close()

    return stock_pos_df



#计算分层指数的相关系数
def compute_stock_layer_factor_index_corr(sf_id):


    all_stocks = stock_util.all_stock_info()
    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index('globalid')
    month_last_trade_dates = stock_factor_util.month_last_day().tolist()

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()


    #获取每期选因子的两端
    sql = session.query(stock_factor_rankcorr.sf_id, stock_factor_rankcorr.trade_date, stock_factor_rankcorr.rankcorr).statement
    rankcorr_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'sf_id'], parse_dates = ['trade_date'])
    rankcorr_df = rankcorr_df.unstack()
    rankcorr_df.columns = rankcorr_df.columns.droplevel(0)


    stock_factor_positive_pos = {}
    stock_factor_negative_pos = {}
    sf_rankcorr = rankcorr_df[sf_id]
    for date in sf_rankcorr.index.tolist():
        rankcorr = sf_rankcorr.loc[date]
        positive_record = session.query(stock_factor_layer.stock_ids).filter(and_(stock_factor_layer.sf_id == sf_id, stock_factor_layer.layer == 0, stock_factor_layer.trade_date == date)).first()
        negative_record = session.query(stock_factor_layer.stock_ids).filter(and_(stock_factor_layer.sf_id == sf_id, stock_factor_layer.layer == 9, stock_factor_layer.trade_date == date)).first()
        if positive_record is None or negative_record is None:
            continue
        stock_factor_positive_pos[date] = json.loads(positive_record[0])
        stock_factor_negative_pos[date] = json.loads(negative_record[0])

    globalid_secode_dict = dict(zip(all_stocks.index.ravel(), all_stocks.sk_secode.ravel()))


    #计算每期股票的仓位
    dates = list(stock_factor_positive_pos.keys())
    dates.sort()
    stock_factor_positive_pos_df = pd.DataFrame(0, index = dates, columns = all_stocks.index)
    stock_factor_negative_pos_df = pd.DataFrame(0, index = dates, columns = all_stocks.index)
    for date in dates:
        positive_stocks = stock_factor_positive_pos[date]
        negative_stocks = stock_factor_negative_pos[date]
        record = stock_factor_positive_pos_df.loc[date]
        record[record.index.isin(positive_stocks)] = 1.0 / len(record.index)
        stock_factor_positive_pos_df.loc[date] = record
        record = stock_factor_negative_pos_df.loc[date]
        record[record.index.isin(negative_stocks)] = 1.0 / len(record.index)
        stock_factor_negative_pos_df.loc[date] = record
    stock_factor_positive_pos_df = stock_factor_positive_pos_df.rename(columns = globalid_secode_dict)
    stock_factor_negative_pos_df = stock_factor_negative_pos_df.rename(columns = globalid_secode_dict)


    #计算因子指数
    caihui_engine = database.connection('caihui')
    caihui_Session = sessionmaker(bind=caihui_engine)
    caihui_session = caihui_Session()

    sql = caihui_session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.secode ,tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.tradedate >= stock_factor_positive_pos_df.index[0].strftime('%Y%m%d')).statement
    stock_yield_df = pd.read_sql(sql, caihui_session.bind, index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']) / 100.0
    stock_yield_df = stock_yield_df.unstack()
    stock_yield_df.columns = stock_yield_df.columns.droplevel(0)
    secodes = list(set(stock_factor_positive_pos_df.columns) & set(stock_yield_df.columns) & set(stock_factor_negative_pos_df.columns))
    stock_factor_positive_pos_df = stock_factor_positive_pos_df[secodes]
    stock_factor_negative_pos_df = stock_factor_negative_pos_df[secodes]
    stock_yield_df = stock_yield_df[secodes]
    caihui_session.commit()
    caihui_session.close()

    stock_factor_positive_pos_df = stock_factor_positive_pos_df.reindex(stock_yield_df.index).fillna(method = 'pad')
    stock_factor_positive_pos_df = stock_factor_positive_pos_df.shift(1).fillna(0.0)

    factor_yield_positive_df = stock_factor_positive_pos_df * stock_yield_df
    factor_yield_positive_df = factor_yield_positive_df.sum(axis = 1)
    factor_yield_positive_df = factor_yield_positive_df[factor_yield_positive_df.index >= '2000-01-01']
    factor_nav_positive_df = (1 + factor_yield_positive_df).cumprod()
    factor_nav_positive_df = factor_nav_positive_df.to_frame()
    factor_nav_positive_df.columns = [sf_id + '_0']


    stock_factor_negative_pos_df = stock_factor_negative_pos_df.reindex(stock_yield_df.index).fillna(method = 'pad')
    stock_factor_negative_pos_df = stock_factor_negative_pos_df.shift(1).fillna(0.0)

    factor_yield_negative_df = stock_factor_negative_pos_df * stock_yield_df
    factor_yield_negative_df = factor_yield_negative_df.sum(axis = 1)
    factor_yield_negative_df = factor_yield_negative_df[factor_yield_negative_df.index >= '2000-01-01']
    factor_nav_negative_df = (1 + factor_yield_negative_df).cumprod()
    factor_nav_negative_df = factor_nav_negative_df.to_frame()
    factor_nav_negative_df.columns = [sf_id + '_9']


    session.commit()
    session.close()

    return pd.concat([factor_nav_positive_df, factor_nav_negative_df], axis = 1)
