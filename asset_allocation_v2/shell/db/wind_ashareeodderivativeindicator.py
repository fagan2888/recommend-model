#coding=utf-8
'''
Created on: May. 14, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import multiprocessing
import numpy as np
import pandas as pd
from . import database
from . import util_db


logger = logging.getLogger(__name__)


def load_a_stock_total_market_value(stock_ids, begin_date=None, end_date=None, reindex=None, fill_method=None):

    stock_ids = util_db.to_list(stock_ids)

    if begin_date is not None:
        begin_date = pd.Timestamp(begin_date)
    if end_date is not None:
        end_date = pd.Timestamp(end_date)

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    res = pool.map(load_a_stock_total_market_value_ser, stock_ids)

    pool.close()
    pool.join()

    df = pd.DataFrame(res, index=stock_ids).T
    df.sort_index(inplace=True)

    if fill_method is not None:
        df.fillna(method=fill_method, inplace=True)
    if begin_date is not None:
        df = df.loc[begin_date:]
    if end_date is not None:
        df = df.loc[:end_date]
    if reindex is not None:
        df = df.reindex(reindex)

    return df

def load_a_stock_total_market_value_ser(stock_id):

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareEODDerivativeIndicator', metadata, autoload=True)

    columns = [
        t.c.TRADE_DT.label('trade_date'),
        t.c.S_VAL_MV.label('total_market_value')
    ]

    s = select(columns).where(t.c.S_INFO_WINDCODE==stock_id)

    df = pd.read_sql(s, engine, index_col=['trade_date'], parse_dates=['trade_date'])
    ser = df.total_market_value.rename(stock_id)

    return ser


if __name__ == '__main__':

    trade_dates = pd.DatetimeIndex(['2019-01-02', '2019-01-03', '2019-01-04'])
    load_a_stock_total_market_value(stock_ids='601318.SH', begin_date=trade_dates[0], end_date=trade_dates[-1], fill_method='pad')
    load_a_stock_total_market_value(stock_ids=['600519.SH', '000568.SZ', '000858.SZ'], reindex=trade_dates)

