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
import stock_util


logger = logging.getLogger(__name__)


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


##过滤掉不合法股票
#def valid_stock_filter(factor_df):
#
#    engine = database.connection('asset')
#    Session = sessionmaker(bind=engine)
#    session = Session()
#
#    for stock_id in factor_df.columns:
#        sql = session.query(stock_factor_stock_valid.trade_date, stock_factor_stock_valid.valid).filter(stock_factor_stock_valid.stock_id == stock_id).statement
#        valid_df = pd.read_sql(sql, session.bind, index_col = ['trade_date'], parse_dates = ['trade_date'])
#        valid_df = valid_df[valid_df.valid == 1]
#        if len(factor_df) == 0:
#            facto_df.stock_id = np.nan
#        else:
#            factor_df[stock_id][~factor_df.index.isin(valid_df.index)] = np.nan
#
#    session.commit()
#    session.close()
#
#    return factor_df


#插入股票合法性表
def compute_stock_valid():

    all_stocks = stock_util.all_stock_info()
    st_stocks = stock_util.stock_st()
    list_date = stock_util.all_stock_listdate()
    list_date.sk_listdate = list_date.sk_listdate + timedelta(365)

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.secode ,tq_qt_skdailyprice.tclose).filter(tq_qt_skdailyprice.secode.in_(all_stocks.index)).statement
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
        logger.info('stock validation date %s done' % date.strftime('%Y-%m-%d'))
    session.commit()
    session.close()

    pass


#过滤掉不合法股票
def valid_stock_filter(factor_df):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    for stock_id in factor_df.columns:
        sql = session.query(stock_factor_stock_valid.trade_date, stock_factor_stock_valid.valid).filter(stock_factor_stock_valid.stock_id == stock_id).statement
        valid_df = pd.read_sql(sql, session.bind, index_col = ['trade_date'], parse_dates = ['trade_date'])
        valid_df = valid_df[valid_df.valid == 1]
        if len(factor_df) == 0:
            facto_df.stock_id = np.nan
        else:
            factor_df[stock_id][~factor_df.index.isin(valid_df.index)] = np.nan

    session.commit()
    session.close()

    logger.info('vailid filter done')

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



#更新因子值数据
def update_factor_value(sf_id, factor_df):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    for stock_id in factor_df.columns:
        ser = factor_df[stock_id].dropna()
        for date in ser.index:
            sfv = stock_factor_value()
            sfv.sf_id = sf_id
            sfv.stock_id = stock_id
            sfv.trade_date = date
            sfv.factor_value = ser.loc[date]
            session.merge(sfv)

        session.commit()

    session.close()
    logger.info('update stock factor value done')
    return



#计算财报因子
def financial_report_data(fr_df, name):

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index('sk_compcode')

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
