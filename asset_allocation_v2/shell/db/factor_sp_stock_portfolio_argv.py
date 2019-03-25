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
    t = Table('sp_stock_portfolio_argv', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.sp_key,
        t.c.sp_value,
        t.c.sp_desc
    ]

    s = select(columns).where(t.c.globalid==portfolio_id)

    df = pd.read_sql(s, engine, index_col=['globalid', 'sp_key'])

    return df

def save(portfolio_id, df_new):

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio_argv', metadata, autoload=True)

    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]

    s = select(columns).where(t.c.globalid==portfolio_id)

    df_old = pd.read_sql(s, engine, index_col=['globalid', 'sp_key'])

    database.batch(engine, t, df_new, df_old, timestamp=True)

