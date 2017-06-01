#coding=utf8

from sqlalchemy import MetaData, Table, select, func, literal_column
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# mz_markowitz
#
# def load(gids, xtypes=None):
#     db = database.connection('asset')
#     metadata = MetaData(bind=db)
#     t1 = Table('mz_markowitz', metadata, autoload=True)

#     columns = [
#         t1.c.globalid,
#         t1.c.mz_type,
#         t1.c.mz_pool,
#         t1.c.mz_reshape,
#         t1.c.mz_name,
#     ]

#     s = select(columns)

#     if gids is not None:
#         s = s.where(t1.c.globalid.in_(gids))
#     if xtypes is not None:
#         s = s.where(t1.c.mz_type.in_(xtypes))
    
#     df = pd.read_sql(s, db)

#     return df
def load_series(gid, reindex=None, begin_date=None, end_date=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('mz_highlow_nav', metadata, autoload=True)

    columns = [
        t1.c.mz_date,
        t1.c.mz_nav,
    ]

    s = select(columns)

    s = s.where(t1.c.mz_highlow_id == gid)

    if begin_date is not None:
        s = s.where(t1.c.mz_date >= begin_date)
        
    if end_date is not None:
        s = s.where(t1.c.mz_date <= end_date)
    
    df = pd.read_sql(s, db, index_col = ['mz_date'], parse_dates=['mz_date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['mz_nav']

def save(gid, df):
    fmt_columns = ['mz_nav', 'mz_inc']
    fmt_precision = 6
    if not df.empty:
        df = database.number_format(df, fmt_columns, fmt_precision)
    #
    # 保存择时结果到数据库
    #
    db = database.connection('asset')
    t2 = Table('mz_highlow_nav', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns, (t2.c.mz_highlow_id == gid))
    df_old = pd.read_sql(s, db, index_col=['mz_highlow_id', 'mz_date'], parse_dates=['mz_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, fmt_columns, fmt_precision)

    # 更新数据库
    # print df_new.head()
    # print df_old.head()
    database.batch(db, t2, df, df_old, timestamp=False)
