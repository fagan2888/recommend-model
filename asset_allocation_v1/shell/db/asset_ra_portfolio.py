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
# ra_portfolio
#
def load(gids, xtypes=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_portfolio', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.ra_type,
        t1.c.ra_algo,
        t1.c.ra_ratio_id,
        t1.c.ra_persistent,
        t1.c.ra_name,
    ]

    s = select(columns)

    if gids is not None:
        s = s.where(t1.c.globalid.in_(gids))
    if xtypes is not None:
        s = s.where(t1.c.ra_type.in_(xtypes))

    df = pd.read_sql(s, db)

    return df

def max_id_between(min_id, max_id):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('ra_portfolio', metadata, autoload=True)

    columns = [ t.c.globalid ]

    # s = select([func.max(t.c.globalid.op('DIV')('10')).label('maxid')]).where(t.c.globalid.between(min_id, max_id))
    s = select([func.max(t.c.globalid).label('maxid')]).where(t.c.globalid.between(min_id, max_id))

    return s.execute().scalar()

def save(gid, df):
    #
    # 保存择时结果到数据库
    #
    db = database.connection('asset')
    t2 = Table('ra_portfolio', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns, (t2.c.globalid == gid))
    df_old = pd.read_sql(s, db, index_col=['globalid'])

    # 更新数据库
    # print df_new.head()
    # print df_old.head()
    database.batch(db, t2, df, df_old, timestamp=False)
