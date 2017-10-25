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
from util.xdebug import dd

logger = logging.getLogger(__name__)

#
# mz_highlow
#
def load(gids):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('mz_highlow_argv', metadata, autoload=True)

    columns = [
        t1.c.mz_highlow_id,
        t1.c.mz_key,
        t1.c.mz_value,
        t1.c.mz_desc,
    ]

    s = select(columns)

    if gids is not None:
        s = s.where(t1.c.mz_highlow_id.in_(gids))

    df = pd.read_sql(s, db, index_col=['mz_highlow_id', 'mz_key'])

    return df

# def max_id_between(min_id, max_id):
#     db = database.connection('asset')
#     metadata = MetaData(bind=db)
#     t = Table('mz_highlow', metadata, autoload=True)

#     columns = [ t.c.globalid ]

#     s = select([func.max(t.c.globalid).label('maxid')]).where(t.c.globalid.between(min_id, max_id))

#     return s.execute().scalar()
def save(gids, df):
    #
    # 保存择时结果到数据库
    #
    db = database.connection('asset')
    t2 = Table('mz_highlow_argv', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns, (t2.c.mz_highlow_id.in_(gids)))
    df_old = pd.read_sql(s, db, index_col=['mz_highlow_id', 'mz_key'])

    # 更新数据库
    # print df_new.head()
    # print df_old.head()
    database.batch(db, t2, df, df_old, timestamp=True)

