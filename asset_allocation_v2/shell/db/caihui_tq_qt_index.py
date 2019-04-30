#coding=utf-8
'''
Modified on: Apr. 12, 2019
Editor: Shixun Su
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


def load_index_daily_data(secode, start_date=None, end_date=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_qt_index', metadata, autoload=True)

    columns = [
        t.c.TRADEDATE.label('date'),
        t.c.TCLOSE.label('close'),
        t.c.THIGH.label('high'),
        t.c.TLOW.label('low'),
        t.c.VOL.label('volume'),
        t.c.TOPEN.label('open'),
    ]

    s = select(columns).where(t.c.SECODE == secode)
    if start_date:
        s = s.where(t.c.TRADEDATE >= start_date)
    if end_date:
        s = s.where(t.c.TRADEDATE <= end_date)
    s = s.where(t.c.ISVALID == 1).order_by(t.c.TRADEDATE.asc())

    df = pd.read_sql(s, engine, index_col = ['date'], parse_dates=['date'])

    return df


def load_index_nav(index_ids, begin_date=None, end_date=None, reindex=None, fill_method=None):

    if isinstance(index_ids, str):
        index_ids = [index_ids]
    elif isinstance(index_ids, (tuple, set)):
        index_ids = list(index_ids)
    elif isinstance(index_ids, dict):
        index_ids = list(index_ids.values())
    else:
        if isinstance(index_ids, (pd.Index, pd.Series, pd.DataFrame)):
            index_ids = index_ids.values
        if isinstance(index_ids, np.ndarray):
            index_ids = index_ids.reshape(-1).tolist()

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    res = pool.map(load_index_nav_ser, index_ids)

    pool.close()
    pool.join()

    df = pd.DataFrame(res, index=index_ids).T
    if fill_method is not None:
        df.fillna(method=fill_method, inplace=True)

    if begin_date is not None:
        df = df.loc[begin_date:]
    if end_date is not None:
        df = df.loc[:end_date]
    if reindex is not None:
        df = df.reindex(reindex)

    return df

def load_index_nav_ser(index_id):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_qt_index', metadata, autoload=True)

    columns = [
        t.c.TRADEDATE.label('trade_date'),
        t.c.TCLOSE.label('nav')
    ]

    s = select(columns).where(t.c.SECODE==index_id)

    df = pd.read_sql(s, engine, index_col=['trade_date'], parse_dates=['trade_date'])
    ser = df.nav.rename(index_id)

    return ser


if __name__ == '__main__':

    load_index_daily_data('2070006540', '20170101', '20170331')
    load_index_nav(index_ids=['2070000005', '2070000014', '2070000553', '2070000060', '2070000187'])

