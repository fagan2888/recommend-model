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


logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def sf(ctx):
    '''multi factor
    '''
    pass


@sf.command()
@click.pass_context
def compute_stock_factor(ctx):
    ln_price_df = ln_price_factor()
    #highlow_price_factor()
    #relative_strength_factor()
    #std_factor()
    #trade_volumn_factor()
    #turn_rate_factor()
    #weighted_strength_factor()
    pass


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


def all_stock_info():

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    all_stocks = pd.read_sql(session.query(ra_stock.globalid, ra_stock.sk_secode).statement, session.bind, index_col = ['sk_secode'])
    session.commit()
    session.close()

    return all_stocks


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


def stock_validate():
    return


#取有因子值的最后一个日期
def stock_factor_last_date(sf_id, stock_id):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    result = session.query(stock_factor_value.trade_date).filter(and_(stock_factor_value.stock_id == stock_id, stock_factor_value.sf_id == sf_id)).order_by(stock_factor_value.trade_date.desc()).first()

    session.commit()
    session.close()

    if result == None:
        return datetime(1900,1,1)
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
    for i in range(0, len(all_stocks.index) - 3400):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(10)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.tclose).filter(and_(tq_qt_skdailyprice.secode == secode, tq_qt_skdailyprice.tradedate >= last_date)).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate']).replace(0.0, np.nan)
        quotation = quotation.sort_index()
        data[globalid] = np.log(quotation.tclose)

    session.commit()
    session.close()

    ln_price_df = pd.DataFrame(data)

    ln_price_df = normalized(ln_price_df)
    update_factor_value(sf_id, ln_price_df)

    return ln_price_df


#股票合法性表
def stock_valid_table():

    pass


#归一化
def normalized(factor_df):

    print factor_df.head()
    factor_mean = factor_df.mean(axis = 1)
    print factor_mean.head()
    factor_std  = factor_df.mean(axis = 1)

    return factor_df

#更新因子值数据
def update_factor_value(sf_id, factor_df):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    for stock_id in factor_df.columns:
        ser = factor_df[stock_id].dropna()
        #ser = ser.where(pd.notnull(ser), None)
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

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_12m = {}
    data_6m = {}
    data_3m = {}
    data_1m = {}

    for i in range(0, len(all_stocks.index) - 3300):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(370)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_sk_dquoteindic.tradedate, tq_sk_dquoteindic.tcloseaf).filter(tq_sk_dquoteindic.secode == secode).filter(tq_sk_dquoteindic.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
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

    update_factor_value('SF.000015', highlow_price_12m_df)
    update_factor_value('SF.000018', highlow_price_6m_df)
    update_factor_value('SF.000017', highlow_price_3m_df)
    update_factor_value('SF.000016', highlow_price_1m_df)

    return


#动量因子
def relative_strength_factor():

    all_stocks = all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_6m = {}
    data_3m = {}
    data_1m = {}
    data    = {}

    for i in range(0, len(all_stocks.index) - 3300):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(10)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.Yield, tq_sk_yieldindic.yieldm, tq_sk_yieldindic.yield3m, tq_sk_yieldindic.yield6m).filter(tq_sk_yieldindic.secode == secode).filter(tq_sk_yieldindic.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
        quotation = quotation.sort_index()
        quotation = quotation / 100.0

        data_6m[globalid] = quotation.yield6m
        data_3m[globalid] = quotation.yield3m
        data_1m[globalid] = quotation.yieldm
        data[globalid] = quotation.Yield

    session.commit()
    session.close()

    relative_strength_6m_df = pd.DataFrame(data_6m)
    relative_strength_3m_df = pd.DataFrame(data_3m)
    relative_strength_1m_df = pd.DataFrame(data_1m)
    relative_strength_df = pd.DataFrame(data_1m)

    return relative_strength_df, relative_strength_1m_df, relative_strength_3m_df, relative_strength_6m_df



#波动率因子
def std_factor():

    all_stocks = all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_6m = {}
    data_3m = {}
    data_1m = {}
    data_12m= {}

    for i in range(0, len(all_stocks.index) - 3400):

        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(370)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode == secode).filter(tq_sk_yieldindic.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
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

    print std_12m_df.tail()

    return std_1m_df, std_3m_df, std_6m_df, std_12m_df


#成交金额因子
def trade_volumn_factor():

    all_stocks = all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_6m = {}
    data_3m = {}
    data_1m = {}
    data_12m= {}

    for i in range(0, len(all_stocks.index) - 3400):

        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(370)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.amount).filter(tq_qt_skdailyprice.secode == secode).filter(tq_qt_skdailyprice.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
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

    print amount_12m_df.tail()

    return amount_1m_df, amount_3m_df, amount_6m_df, amount_12m_df


#换手率因子
def turn_rate_factor():

    all_stocks = all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_6m = {}
    data_3m = {}
    data_1m = {}
    data_12m= {}

    for i in range(0, len(all_stocks.index) - 3400):

        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(10)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.turnrate, tq_sk_yieldindic.turnratem, tq_sk_yieldindic.turnrate3m,
                tq_sk_yieldindic.turnrate6m, tq_sk_yieldindic.turnratey).filter(tq_sk_yieldindic.secode == secode).filter(tq_sk_yieldindic.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
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

    print turnrate_1m_df.tail()

    return turnrate_1m_df, turnrate_3m_df, turnrate_6m_df, turnrate_12m_df


#加权动量因子
def weighted_strength_factor():

    all_stocks = all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_6m = {}
    data_3m = {}
    data_1m = {}
    data_12m= {}

    for i in range(0, len(all_stocks.index) - 3400):

        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print globalid, secode
        last_date = stock_factor_last_date(sf_id, globalid) - timedelta(370)
        last_date = last_date.strftime('%Y%m%d')
        sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.turnrate, tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode == secode).filter(tq_sk_yieldindic.tradedate >= last_date).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
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

    return weighted_strength_1m_df, weighted_strength_3m_df, weighted_strength_6m_df, weighted_strength_12m_df
