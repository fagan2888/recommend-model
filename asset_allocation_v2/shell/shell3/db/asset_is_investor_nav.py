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
# is_markowitz
#
# def load(gids, xtypes=None):
#     db = database.connection('asset')
#     metadata = MetaData(bind=db)
#     t1 = Table('is_markowitz', metadata, autoload=True)

#     columns = [
#         t1.c.globalid,
#         t1.c.is_type,
#         t1.c.is_pool,
#         t1.c.is_reshape,
#         t1.c.is_name,
#     ]

#     s = select(columns)

#     if gids is not None:
#         s = s.where(t1.c.globalid.in_(gids))
#     if xtypes is not None:
#         s = s.where(t1.c.is_type.in_(xtypes))
    
#     df = pd.read_sql(s, db)

#     return df

def save(gid, xtype, df):
    fmt_columns = ['is_nav', 'is_inc']
    fmt_precision = 6
    if not df.empty:
        df = database.number_format(df, fmt_columns, fmt_precision)
    #
    # 保存择时结果到数据库
    #
    db = database.connection('asset')
    t2 = Table('is_investor_nav', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns, (t2.c.is_investor_id == gid)).where(t2.c.is_type == xtype)
    df_old = pd.read_sql(s, db, index_col=['is_investor_id', 'is_type', 'is_date'], parse_dates=['is_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, fmt_columns, fmt_precision)

    # 更新数据库
    # print df_new.head()
    # print df_old.head()
    database.batch(db, t2, df, df_old, timestamp=True)
