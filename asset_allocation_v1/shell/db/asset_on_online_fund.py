#coding=utf8

# import pdb
from sqlalchemy import MetaData, Table, select, func, literal_column
# import string
# from datetime import datetime, timedelta
import pandas as pd
import os
import sys
import logging
import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

# def load(gid, included_online_id=False):
#     db = database.connection('asset')
#     metadata = MetaData(bind=db)
#     t1 = Table('on_online_pos', metadata, autoload=True)

#     columns = [
#         t1.c.on_date,
#         t1.c.on_asset_id,
#         t1.c.on_ratio,
#     ]
#     index_col = ['on_date', 'on_asset_id']
    
#     if included_online_id:
#         columns.insert(0, t1.c.on_online_id)
#         index_col.insert(0, 'on_online_id')

#     s = select(columns)

#     if gid is not None:
#         s = s.where(t1.c.on_online_id == gid)
#     else:
#         return None
#     # if xtypes is not None:
#     #     s = s.where(t1.c.on_type.in_(xtypes))
    
#     df = pd.read_sql(s, db, index_col=index_col, parse_dates=['on_date'])

#     df = df.unstack().fillna(0.0)
#     df.columns = df.columns.droplevel(0)

#     return df

def load_fund_pos(gid):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('on_online_fund', metadata, autoload=True)

    columns = [
        t1.c.on_date,
        t1.c.on_pool_id,
        t1.c.on_fund_id,
        t1.c.on_fund_ratio,
    ]
    index_col = ['on_date', 'on_pool_id', 'on_fund_id']
    
    # if included_online_id:
    #     columns.insert(0, t1.c.on_online_id)
    #     index_col.insert(0, 'on_online_id')

    s = select(columns)

    if gid is not None:
        s = s.where(t1.c.on_online_id == gid)
    else:
        return None
    # if xtypes is not None:
    #     s = s.where(t1.c.on_type.in_(xtypes))
    
    df = pd.read_sql(s, db, index_col=index_col, parse_dates=['on_date'])

    #
    # 合并来自不同基金池的基金份额
    #
    df_result = df.groupby(level=['on_date', 'on_fund_id']).sum()

    print df_result
    df_result = df_result.loc[df_result['on_fund_ratio'] > 0]

    # df_result = df_result.unstack().fillna(0.0)
    # df_result.columns = df_result.columns.droplevel(0)

    return df_result


# def save(gid, df):
#     fmt_columns = ['on_fund_ratio']
#     fmt_precision = 4
#     if not df.empty:
#         df['on_fund_type'] = df['on_fund_type'].astype(int)
#         df = database.number_format(df, fmt_columns, fmt_precision)
#     #
#     # 保存择时结果到数据库
#     #
#     db = database.connection('asset')
#     t2 = Table('on_online_pos', MetaData(bind=db), autoload=True)
#     columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
#     s = select(columns, (t2.c.on_online_id == gid))
#     df_old = pd.read_sql(s, db, index_col=['on_online_id', 'on_date', 'on_pool_id', 'on_fund_id'], parse_dates=['on_date'])
#     if not df_old.empty:
#         df_old = database.number_format(df_old, fmt_columns, fmt_precision)

#     # 更新数据库
#     # print df_new.head()
#     # print df_old.head()
#     database.batch(db, t2, df, df_old, timestamp=True)

