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
from . import database
from . import util_db


logger = logging.getLogger(__name__)


def load_a_index_nav(index_ids, begin_date=None, end_date=None, reindex=None, fill_method=None):

    index_ids = util_db.to_list(index_ids)

    if begin_date is not None:
        begin_date = pd.Timestamp(begin_date)
    if end_date is not None:
        end_date = pd.Timestamp(end_date)

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    res = pool.map(load_a_index_nav_ser, index_ids)

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

def load_a_index_nav_ser(index_id):

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AIndexEODPrices', metadata, autoload=True)

    columns = [
        t.c.TRADE_DT.label('trade_date'),
        t.c.S_DQ_CLOSE.label('nav')
    ]

    s = select(columns).where(t.c.S_INFO_WINDCODE==index_id)

    df = pd.read_sql(s, engine, index_col=['trade_date'], parse_dates=['trade_date'])
    ser = df.nav.sort_index().rename(index_id)

    return ser


if __name__ == '__main__':

    load_a_index_nav(index_ids=['000300.SH', '000905.SH'])

