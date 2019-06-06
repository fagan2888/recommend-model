#coding=utf-8
'''
Created on: May. 8, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import multiprocessing
import numpy as np
import pandas as pd
import functools
# from ipdb import set_trace
# from time import perf_counter
from . import database
from . import util_db


logger = logging.getLogger(__name__)


def load_a_stock_adj_price(stock_ids, begin_date=None, end_date=None, reindex=None, fill_method=None):

    stock_ids = util_db.to_list(stock_ids)

    if begin_date is not None:
        begin_date = pd.Timestamp(begin_date)
    if end_date is not None:
        end_date = pd.Timestamp(end_date)

    # ser = pd.Series(stock_ids, index=stock_ids)
    # rrr = ser.apply(load_a_stock_adj_price_ser)
    # set_trace()

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    res = pool.map(load_a_stock_adj_price_ser, stock_ids)

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

def load_a_stock_adj_price_ser(stock_id):

    # start =perf_counter()

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareEODPrices', metadata, autoload=True)

    columns = [
        t.c.TRADE_DT.label('trade_date'),
        t.c.S_DQ_ADJCLOSE.label('adj_prc')
    ]

    s = select(columns).where(t.c.S_INFO_WINDCODE==stock_id)

    df = pd.read_sql(s, engine, index_col=['trade_date'], parse_dates=['trade_date'])
    ser = df.adj_prc.sort_index().rename(stock_id)

    # print(perf_counter()-start)

    return ser

def load_a_stock_price(stock_ids, begin_date=None, end_date=None, reindex=None, fill_method=None):

    stock_ids = util_db.to_list(stock_ids)

    if begin_date is not None:
        begin_date = pd.Timestamp(begin_date)
    if end_date is not None:
        end_date = pd.Timestamp(end_date)

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    res = pool.map(load_a_stock_price_ser, stock_ids)

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

def load_a_stock_price_ser(stock_id):

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareEODPrices', metadata, autoload=True)

    columns = [
        t.c.TRADE_DT.label('trade_date'),
        t.c.S_DQ_CLOSE.label('prc')
    ]

    s = select(columns).where(t.c.S_INFO_WINDCODE==stock_id)

    df = pd.read_sql(s, engine, index_col=['trade_date'], parse_dates=['trade_date'])
    ser = df.prc.sort_index().rename(stock_id)

    return ser

def load_a_stock_status(stock_ids, begin_date=None, end_date=None, reindex=None):

    stock_ids = util_db.to_list(stock_ids)

    if reindex is not None:

        reindex_sorted = reindex.sort_values()
        if begin_date is None:
            begin_date = reindex_sorted[0].strftime('%Y%m%d')
        if end_date is None:
            end_date = reindex_sorted[-1].strftime('%Y%m%d')

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    kwargs = {'begin_date': begin_date, 'end_date': end_date}
    res = pool.map(functools.partial(load_a_stock_status_ser, **kwargs), stock_ids)

    pool.close()
    pool.join()

    df = pd.DataFrame(res, index=stock_ids).T
    df.fillna(4, inplace=True)

    if reindex is not None:
        df = df.reindex(reindex, method=None)

    return df

def load_a_stock_status_ser(stock_id, begin_date=None, end_date=None):

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareEODPrices', metadata, autoload=True)

    columns = [
        t.c.TRADE_DT.label('trade_date'),
        t.c.S_DQ_PRECLOSE.label('l_close'),
        t.c.S_DQ_CLOSE.label('t_close'),
        t.c.S_DQ_VOLUME.label('vol')
    ]

    s = select(columns).where(t.c.S_INFO_WINDCODE==stock_id)
    if begin_date is not None:
        s = s.where(t.c.TRADE_DT>=begin_date)
    if end_date is not None:
        s = s.where(t.c.TRADE_DT<=end_date)

    df = pd.read_sql(s, engine, index_col=['trade_date'], parse_dates=['trade_date'])
    if df.size > 0:
        ser = df.apply(status_algo, axis='columns').sort_index().rename(stock_id)
    else:
        ser = pd.Series(name=stock_id)

    return ser

def status_algo(ser):

    if ser.loc['vol'] == 0:
        return 3
    elif round(ser.loc['l_close']*1.1, 2) <= ser.loc['t_close']:
        return 1
    elif round(ser.loc['l_close']*0.9, 2) >= ser.loc['t_close']:
        return 2
    else:
        return 0


if __name__ == '__main__':

    trade_dates = pd.DatetimeIndex(['2019-01-02', '2019-01-03', '2019-01-04'])
    load_a_stock_adj_price(stock_ids='601318.SH', begin_date=trade_dates[0], end_date=trade_dates[-1], fill_method='pad')
    load_a_stock_adj_price(stock_ids=['600519.SH', '000568.SZ', '000858.SZ'], reindex=trade_dates)
    load_stock_price(stock_ids={'600270.SH', '601598.SH'})

