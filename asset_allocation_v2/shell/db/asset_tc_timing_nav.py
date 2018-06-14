#coding=utf8

from sqlalchemy import MetaData, Table, select, func, literal_column
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
from . import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# tc_markowitz
#
# def load(gids, xtypes=None):
#     db = database.connection('asset')
#     metadata = MetaData(bind=db)
#     t1 = Table('tc_markowitz', metadata, autoload=True)

#     columns = [
#         t1.c.globalid,
#         t1.c.tc_type,
#         t1.c.tc_pool,
#         t1.c.tc_reshape,
#         t1.c.tc_name,
#     ]

#     s = select(columns)

#     if gids is not None:
#         s = s.where(t1.c.globalid.in_(gids))
#     if xtypes is not None:
#         s = s.where(t1.c.tc_type.in_(xtypes))
    
#     df = pd.read_sql(s, db)

#     return df

def load_series(gid, reindex=None, begin_date=None, end_date=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('tc_timing_nav', metadata, autoload=True)

    columns = [
        t1.c.tc_date,
        t1.c.tc_nav,
    ]

    s = select(columns)

    s = s.where(t1.c.tc_timing_id == gid)

    if begin_date is not None:
        s = s.where(t1.c.tc_date >= begin_date)
        
    if end_date is not None:
        s = s.where(t1.c.tc_date <= end_date)
    
    df = pd.read_sql(s, db, index_col = ['tc_date'], parse_dates=['tc_date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['tc_nav']


# def save(gid, xtype, df):
#     fmt_columns = ['tc_nav', 'tc_inc']
#     fmt_precision = 6
#     if not df.empty:
#         df = database.number_format(df, fmt_columns, fmt_precision)
#     #
#     # 保存择时结果到数据库
#     #
#     db = database.connection('asset')
#     t2 = Table('tc_timing_nav', MetaData(bind=db), autoload=True)
#     columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
#     s = select(columns, (t2.c.tc_timing_id == gid)).where(t2.c.tc_type == xtype)
#     df_old = pd.read_sql(s, db, index_col=['tc_timing_id', 'tc_type', 'tc_date'], parse_dates=['tc_date'])
#     if not df_old.empty:
#         df_old = database.number_format(df_old, fmt_columns, fmt_precision)

#     # 更新数据库
#     # print df_new.head()
#     # print df_old.head()
#     database.batch(db, t2, df, df_old, timestamp=True)
