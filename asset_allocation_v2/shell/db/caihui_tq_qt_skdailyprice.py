#coding=utf-8
'''
Created on: Mar. 24, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import multiprocessing
import functools
import numpy as np
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_stock_status(stock_ids, begin_date=None, end_date=None, reindex=None):

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

    if reindex is not None:

        reindex_sorted = reindex.sort_values()
        if begin_date is None:
            begin_date = reindex_sorted[0].strftime('%Y%m%d')
        if end_date is None:
            end_date = reindex_sorted[-1].strftime('%Y%m%d')

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    kwargs = {'begin_date': begin_date, 'end_date': end_date}
    res = pool.map(functools.partial(load_stock_status_ser, **kwargs), stock_ids)

    pool.close()
    pool.join()

    df = pd.DataFrame(res, index=stock_ids).T
    df.fillna(4, inplace=True)

    if reindex is not None:
        df = df.reindex(reindex, method=None)

    return df

def load_stock_status_ser(stock_id, begin_date=None, end_date=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_qt_skdailyprice', metadata, autoload=True)

    columns = [
        t.c.TRADEDATE.label('trade_date'),
        t.c.LCLOSE.label('l_close'),
        t.c.TCLOSE.label('t_close'),
        t.c.VOL.label('vol')
    ]

    s = select(columns).where(t.c.SECODE==stock_id)
    if begin_date is not None:
        s = s.where(t.c.TRADEDATE>=begin_date)
    if end_date is not None:
        s = s.where(t.c.TRADEDATE<=end_date)

    df = pd.read_sql(s, engine, index_col=['trade_date'], parse_dates=['trade_date'])
    ser = df.apply(status_algo, axis='columns').rename(stock_id)

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

    load_stock_price(stock_ids={'2010000001', '2010000005'})

