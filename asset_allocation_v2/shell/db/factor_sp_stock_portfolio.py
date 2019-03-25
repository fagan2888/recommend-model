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


def load_all():

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.sp_type,
        t.c.sp_algo,
        t.c.sp_name
    ]

    s = select(columns)

    df = pd.read_sql(s, engine, index_col=['globalid'])

    return df

def load_by_id(portfolio_ids):

    if isinstance(portfolio_ids, str):
        portfolio_ids = [portfolio_ids]

    res = []
    for portfolio_id in portfolio_ids:
        res.append(load_by_id_df(portfolio_id))

    df = pd.concat(res)

    return df

def load_by_id_df(portfolio_id):

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.sp_type,
        t.c.sp_algo,
        t.c.sp_name
    ]

    s = select(columns).where(t.c.globalid.like(f'{portfolio_id}%%'))

    df = pd.read_sql(s, engine, index_col=['globalid'])

    return df

def load_by_type(types):

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.sp_type,
        t.c.sp_algo,
        t.c.sp_name
    ]

    s = select(columns).where(t.c.sp_type.in_(types))

    df = pd.read_sql(s, engine, index_col=['globalid'])

    return df

def max_id_between(min_id, max_id):

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio', metadata, autoload=True)

    columns = [
        t.c.globalid
    ]

    s = select([func.max(t.c.globalid)]).where(t.c.globalid.between(min_id, max_id))

    return s.execute().scalar()

def save(portfolio_id, df_new):

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    t = Table('sp_stock_portfolio', metadata, autoload=True)

    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]

    s = select(columns).where(t.c.globalid==portfolio_id)

    df_old = pd.read_sql(s, engine, index_col=['globalid'])

    database.batch(engine, t, df_new, df_old, timestamp=True)

