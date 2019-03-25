#coding=utf-8
'''
Created on: Mar. 20, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func, literal_column
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_nav(portfolio_id, begin_date=None, end_date=None, reindex=None):

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio_nav', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.sp_date,
        t.c.sp_nav,
    ]

    s = select(columns).where(t.c.globalid==portfolio_id)
    if begin_date is not None:
        s = s.where(t.c.sp_date>=begin_date)
    if end_date is not None:
        s = s.where(t.c.sp_date<=end_date)

    df = pd.read_sql(s, engine, parse_dates=['sp_date'])

    df = df.pivot('sp_date', 'globalid', 'sp_nav')
    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df

def save(portfolio_id, df_new):

    fmt_columns = ['sp_nav', 'sp_inc']
    fmt_precision = 6
    if not df_new.empty:
        df_new = database.number_format(df_new, fmt_columns, fmt_precision)

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio_nav', metadata, autoload=True)

    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]

    s = select(columns).where(t.c.globalid==portfolio_id)

    df_old = pd.read_sql(s, engine, index_col=['globalid', 'sp_date'], parse_dates=['sp_date'])

    if not df_old.empty:
        df_old = database.number_format(df_old, fmt_columns, fmt_precision)

    database.batch(engine, t, df_new, df_old, timestamp=True)

