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


def load(portfolio_id):

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio_pos', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.sp_date,
        t.c.sp_sk_id,
        t.c.sp_sk_pos
    ]

    s = select(columns).where(t.c.globalid==portfolio_id)

    df = pd.read_sql(s, engine, index_col=['globalid', 'sp_date', 'sp_sk_id'], parse_dates=['sp_date'])

    df = df.sp_sk_pos.unstack()

    return df

def save(portfolio_id, df_new):

    fmt_columns = ['sp_sk_pos']
    fmt_precision = 6
    if not df_new.empty:
        df_new = database.number_format(df_new, fmt_columns, fmt_precision)

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio_pos', metadata, autoload=True)

    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]

    s = select(columns).where(t.c.globalid==portfolio_id)

    df_old = pd.read_sql(s, engine, index_col=['globalid', 'sp_date', 'sp_sk_id'], parse_dates=['sp_date'])

    if not df_old.empty:
        df_old = database.number_format(df_old, fmt_columns, fmt_precision)

    database.batch(engine, t, df_new, df_old, timestamp=True)

