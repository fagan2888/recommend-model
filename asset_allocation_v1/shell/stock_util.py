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
from db.asset_stock_factor import *
from db.asset_stock import *
from db import asset_trade_dates
import math
import json


logger = logging.getLogger(__name__)


#所有股票代码
def all_stock_info():

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    all_stocks = pd.read_sql(session.query(ra_stock.globalid, ra_stock.sk_secode, ra_stock.sk_compcode, ra_stock.sk_name).statement, session.bind, index_col = ['sk_secode'])
    session.commit()
    session.close()

    return all_stocks


#所有股票上市日期
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


#根据每期股票等权配置每期股票权重
def stock_pos_2_weight(stock_pos):

    all_stocks = all_stock_info()

    dates = list(stock_pos.keys())
    dates.sort()
    datas = {}
    for date in dates:
        stocks = stock_pos[date]
        stocks = list(stocks)
        stocks_num = 1.0 * len(stocks)
        record = pd.Series(0, index = all_stocks.globalid)
        record[record.index.isin(stocks)] = 1.0 / stocks_num
        #stock_pos_df.loc[date] = record
        #for stock_id in stocks:
        #    record.loc[stock_id] = 1.0 * stocks.count(stock_id) / stocks_num
        datas[date] = record

    stock_pos_df = pd.DataFrame(datas).T
    return stock_pos_df



#根据每期股票按照因子权重配置每期股票权重
def stock_pos_2_factor_weight(bf_id, stock_pos):

    all_stocks = all_stock_info()

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    dates = list(stock_pos.keys())
    dates.sort()
    datas = {}

    for date in dates:
        stocks = stock_pos[date]
        stocks = list(set(stocks) & set(all_stocks.globalid.ravel()))
        sql = session.query(barra_stock_factor_exposure.stock_id, barra_stock_factor_exposure.factor_exposure).filter(and_(barra_stock_factor_exposure.bf_id == bf_id, barra_stock_factor_exposure.trade_date == date)).statement
        factor_exposure_df = pd.read_sql(sql, session.bind, index_col = ['stock_id'])
        factor_exposure_df = factor_exposure_df.loc[stocks]
        factor_exposure_df = factor_exposure_df / factor_exposure_df.sum()
        record = pd.Series(0, index = all_stocks.globalid)
        for stock_id in factor_exposure_df.index:
            record.loc[stock_id] = factor_exposure_df.loc[stock_id].ravel()[0]
        datas[date] = record

    stock_pos_df = pd.DataFrame(datas).T

    return stock_pos_df