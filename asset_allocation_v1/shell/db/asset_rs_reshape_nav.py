#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

def load(gids):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('rs_reshape_pos', metadata, autoload=True)

    columns = [
        t1.c.rs_reshape_id,
        t1.c.rs_date,
        t1.c.rs_ratio,
    ]

    s = select(columns)

    if gids is not None:
        s = s.where(t1.c.rs_reshape_id.in_(gids))
    
    df = pd.read_sql(s, db, index_col = ['rs_date', 'rs_reshape_id'], parse_dates=['rs_date'])

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df

# def load(gids, xtypes=None):
#     db = database.connection('asset')
#     metadata = MetaData(bind=db)
#     t1 = Table('rs_reshape_pos', metadata, autoload=True)

#     columns = [
#         t1.c.globalid,
#         t1.c.rs_type,
#         t1.c.rs_pool,
#         t1.c.rs_name,
#     ]

#     s = select(columns)

#     if gids is not None:
#         s = s.where(t1.c.globalid.in_(gids))
#     if xtypes is not None:
#         s = s.where(t1.c.rs_type.in_(xtypes))
    
#     df = pd.read_sql(s, db)

#     return df

def load_series(gid, reindex=None, begin_date=None, end_date=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('rs_reshape_nav', metadata, autoload=True)

    columns = [
        t1.c.rs_date,
        t1.c.rs_nav,
    ]

    s = select(columns)

    if gid is not None:
        s = s.where(t1.c.rs_reshape_id == gid)
    if begin_date is not None:
        s = s.where(t1.c.rs_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.rs_date <= end_date)
    
    df = pd.read_sql(s, db, index_col = ['rs_date'], parse_dates=['rs_date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['rs_nav']

