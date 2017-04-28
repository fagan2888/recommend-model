#coding=utf8

from sqlalchemy import MetaData, Table, select, func, literal_column
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database
import asset_ra_pool

from dateutil.parser import parse

logger = logging.getLogger(__name__)

def load(gid, reindex=None):
    pool = asset_ra_pool.find(gid)
    if pool:
        (pool_id, category) = (gid, 0)
    else:
        asset_id = gid % 10000000
        (pool_id, category) = (asset_id / 100, asset_id % 100)

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_pool_fund', metadata, autoload=True)

    columns = [
        # t1.c.ra_category,
        t1.c.ra_date,
        t1.c.ra_fund_type,
        t1.c.ra_fund_id,
        t1.c.ra_fund_code,
        t1.c.ra_fund_level,
        t1.c.ra_sharpe,
        t1.c.ra_jensen,
    ]
    index_col = ['ra_date', 'ra_fund_id']
    
    s = select(columns)

    if gid is not None:
        s = s.where(t1.c.ra_pool == pool_id).where(t1.c.ra_category == category)
    else:
        return None
    # if xtypes is not None:
    #     s = s.where(t1.c.mz_type.in_(xtypes))
    
    df = pd.read_sql(s, db, index_col=index_col, parse_dates=['ra_date'])

    if reindex is not None:
        df = df.unstack().reindex(reindex, method='pad').stack()

    # df = df.unstack().fillna(0.0)
    # df.columns = df.columns.droplevel(0)
    df['ra_fund_type'] = tlsFundType(gid)

    return df

def tlsFundType(gid, default=0):
    tls = {
        11210111 : 11101, # 大盘
        11210112 : 11102, # 小盘
        11210113 : 11102, # 上涨
        11210114 : 11102, # 震荡
        11210115 : 11102, # 下跌
        11210116 : 11102, # 成长
        11210117 : 11101, # 价值
        
        11220121 : 12101, # 利率债
        11220122 : 12102, # 信用债
        11220123 : 12103, # 可转债
        
        11230131 : 13101, # 货币
        
        11240141 : 11202, # 标普
        11240142 : 14001, # 黄金
        11240143 : 11205, # 恒生

        19210111 : 11101, # 大盘
        19210112 : 11102, # 小盘
        19210113 : 11102, # 上涨
        19210114 : 11102, # 震荡
        19210115 : 11102, # 下跌
        19210116 : 11102, # 成长
        19210117 : 11101, # 价值
        
        19220121 : 12101, # 利率债
        19220122 : 12102, # 信用债
        19220123 : 12103, # 可转债
        
        19230131 : 13101, # 货币
        11310100 : 13101, # 货币
        
        19240141 : 11202, # 标普
        19240142 : 14001, # 黄金
        19240143 : 11205, # 恒生
    }        

    if gid in tls:
        return tls[gid]
    else:
        return int((gid % 10000000) / 100)


# def save(gid, df):
#     fmt_columns = ['mz_ratio']
#     fmt_precision = 4
#     if not df.empty:
#         df = database.number_format(df, fmt_columns, fmt_precision)
#     #
#     # 保存择时结果到数据库
#     #
#     db = database.connection('asset')
#     t2 = Table('mz_highlow_pos', MetaData(bind=db), autoload=True)
#     columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
#     s = select(columns, (t2.c.mz_highlow_id == gid))
#     df_old = pd.read_sql(s, db, index_col=['mz_highlow_id', 'mz_date', 'mz_asset_id'], parse_dates=['mz_date'])
#     if not df_old.empty:
#         df_old = database.number_format(df_old, fmt_columns, fmt_precision)

#     # 更新数据库
#     # print df_new.head()
#     # print df_old.head()
#     database.batch(db, t2, df, df_old, timestamp=True)

