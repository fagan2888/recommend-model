#coding=utf-8
'''
Created on: Dec. 28, 2018
Modified on: May. 1, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import multiprocessing
import numpy as np
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_stock_price(stock_ids, begin_date=None, end_date=None, reindex=None, fill_method=None):

    if isinstance(stock_ids, str):
        stock_ids = [stock_ids]
    elif isinstance(stock_ids, (tuple, set)):
        stock_ids = list(stock_ids)
    elif isinstance(stock_ids, dict):
        stock_ids = list(stock_ids.values())
    else:
        if isinstance(stock_ids, (pd.Index, pd.Series, pd.DataFrame)):
            stock_ids = stock_ids.values
        if isinstance(stock_ids, np.ndarray):
            stock_ids = stock_ids.reshape(-1).tolist()

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    res = pool.map(load_stock_price_ser, stock_ids)

    pool.close()
    pool.join()

    df = pd.DataFrame(res, index=stock_ids).T
    if fill_method is not None:
        df.fillna(method=fill_method, inplace=True)

    if begin_date is not None:
        df = df.loc[begin_date:]
    if end_date is not None:
        df = df.loc[:end_date]
    if reindex is not None:
        df = df.reindex(reindex)

    return df

def load_stock_price_ser(stock_id):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_sk_dquoteindic', metadata, autoload=True)

    columns = [
        t.c.TRADEDATE.label('trade_date'),
        t.c.TCLOSEAF.label('prc')
    ]

    s = select(columns).where(t.c.SECODE==stock_id)

    df = pd.read_sql(s, engine, index_col=['trade_date'], parse_dates=['trade_date'])
    ser = df.prc.rename(stock_id)

    return ser

def load_stock_market_data(stock_ids, begin_date=None, end_date=None, reindex=None, fill_method=None):

    if isinstance(stock_ids, str):
        stock_ids = [stock_ids]
    elif isinstance(stock_ids, (tuple, set)):
        stock_ids = list(stock_ids)
    elif isinstance(stock_ids, dict):
        stock_ids = list(stock_ids.values())
    else:
        if isinstance(stock_ids, (pd.Index, pd.Series, pd.DataFrame)):
            stock_ids = stock_ids.values
        if isinstance(stock_ids, np.ndarray):
            stock_ids = stock_ids.reshape(-1).tolist()

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    res = pool.map(load_stock_market_data_df, stock_ids)

    pool.close()
    pool.join()

    df = pd.concat(res).unstack()
    if fill_method is not None:
        df.fillna(method=fill_method, inplace=True)

    if begin_date is not None:
        df = df.loc[begin_date:]
    if end_date is not None:
        df = df.loc[:end_date]
    if reindex is not None:
        df = df.reindex(reindex)

    return df

def load_stock_market_data_df(stock_id):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_sk_dquoteindic', metadata, autoload=True)

    columns = [
        t.c.SECODE.label('stock_id'),
        t.c.TRADEDATE.label('trade_date'),
        t.c.TCLOSEAF.label('price'),
        t.c.VOL.label('volume'),
        t.c.AMOUNT.label('amount'),
        t.c.MKTSHARE.label('market_share'),
        t.c.TOTALSHARE.label('total_share')
    ]

    s = select(columns).where(t.c.SECODE==stock_id)

    df = pd.read_sql(s, engine, index_col=['trade_date', 'stock_id'], parse_dates=['trade_date'])

    return df


if __name__ == '__main__':

    load_stock_price(stock_ids='2010000001', begin_date='20181229', end_date='20180107')
    trade_dates = pd.DatetimeIndex(['2019-01-02', '2019-01-03', '2019-01-04'])
    load_stock_market_data(stock_ids=['2010000001', '2010000005', '2010000007'], reindex=trade_dates)

