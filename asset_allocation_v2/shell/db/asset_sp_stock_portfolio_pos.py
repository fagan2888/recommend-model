#coding=utf-8
'''
Created on: Mar. 20, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func, literal_column
import multiprocessing
import pandas as pd
from . import database
from . import util_db


logger = logging.getLogger(__name__)


def load_portfolio_pos(portfolio_ids, begin_date=None, end_date=None, reindex=None):

    portfolio_ids = util_db.to_list(portfolio_ids)

    if begin_date is not None:
        begin_date = pd.Timestamp(begin_date)
    if end_date is not None:
        end_date = pd.Timestamp(end_date)

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    res = pool.map(load_portfolio_pos_ser, portfolio_ids)

    pool.close()
    pool.join()

    df = pd.concat(res)

    return df

def load_portfolio_pos_ser(portfolio_id):

    engine = database.connection('asset')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio_pos', metadata, autoload=True)

    columns = [
        t.c.globalid.label('portfolio_id'),
        t.c.sp_date.label('trade_date'),
        t.c.sp_sk_id.label('stock_id'),
        t.c.sp_sk_pos.label('pos')
    ]

    s = select(columns).where(t.c.globalid==portfolio_id)

    df = pd.read_sql(s, engine, index_col=['portfolio_id', 'trade_date', 'stock_id'], parse_dates=['trade_date'])

    return df

def save(portfolio_id, df_new):

    fmt_columns = ['sp_sk_pos']
    fmt_precision = 6
    if not df_new.empty:
        df_new = database.number_format(df_new, fmt_columns, fmt_precision)

    engine = database.connection('asset')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio_pos', metadata, autoload=True)

    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]

    s = select(columns).where(t.c.globalid==portfolio_id)

    df_old = pd.read_sql(s, engine, index_col=['globalid', 'sp_date', 'sp_sk_id'], parse_dates=['sp_date'])

    if not df_old.empty:
        df_old = database.number_format(df_old, fmt_columns, fmt_precision)

    database.batch(engine, t, df_new, df_old, timestamp=True)

