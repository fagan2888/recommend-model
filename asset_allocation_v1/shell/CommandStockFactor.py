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
    '''
    factor_funcs = [ln_capital_factor, ln_price_factor, highlow_price_factor, relative_strength_factor, std_factor, trade_volumn_factor, turn_rate_factor, weighted_strength_factor]
    pool = Pool(7)
    pool.map(call_factor_func, factor_funcs)
    pool.close()
    pool.join()
    '''

    highlow_price_factor()

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
    all_stocks = pd.read_sql(session.query(ra_stock.globalid, ra_stock.sk_secode).statement, session.bind, index_col = ['sk_secode'])
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


#bp因子
def bp_factor():

    all_stocks = all_stock_info()
    sf_id = 'SF.000005'



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

    '''
    sf_ids = ['SF.000001','SF.000002', 'SF.000015', 'SF.000016','SF.000017','SF.000018', 'SF.000035', 'SF.000036', 'SF.000037', 'SF.000038', 
            'SF.000047','SF.000048','SF.000049','SF.000050','SF.000051','SF.000052','SF.000053','SF.000054','SF.000055','SF.000056',
            'SF.000057','SF.000058','SF.000059','SF.000060','SF.000061','SF.000062',
            ]


    pool = Pool(10)
    pool.map(stock_factor_layer_spearman, sf_ids)
    pool.close()
    pool.join()
    '''

    stock_factor_layer_spearman('SF.000001')

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
            'SF.000057','SF.000058','SF.000059','SF.000060','SF.000061','SF.000062',
            ]


    pool = Pool(16)
    pool.map(stock_factor_index, sf_ids)
    pool.close()
    pool.join()

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

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(stock_factor_rankcorr.sf_id, stock_factor_rankcorr.trade_date, stock_factor_rankcorr.rankcorr).statement
    rankcorr_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'sf_id'], parse_dates = ['trade_date'])
    rankcorr_df = rankcorr_df.unstack()
    rankcorr_df.columns = rankcorr_df.columns.droplevel(0)

    rankcorr_df = rankcorr_df.rolling(14).mean()
    rankcorr_abs_df = abs(rankcorr_df)

    for date in rankcorr_abs_df.index:
        rankcorr_abs = rankcorr_abs_df.loc[date]
        rankcorr_abs = rankcorr_abs.sort_values(ascending = False)
        for index in rankcorr_abs.index[0:5]:
            rankcorr = rankcorr_df.loc[date, index]
            if rankcorr >= 0:
                print date, index, 0
            else:
                print date, index, 4

    #for i in df.tail(1):
    #    print i, df.tail(1)[i]

    return
