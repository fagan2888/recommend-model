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
from util.xdebug import dd

logger = logging.getLogger(__name__)

#
# ra_portfolio
#
def load(gids):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_portfolio_argv', metadata, autoload=True)

    columns = [
        t1.c.ra_portfolio_id,
        t1.c.ra_key,
        t1.c.ra_value,
        t1.c.ra_desc,
    ]

    s = select(columns)

    if gids is not None:
        s = s.where(t1.c.ra_portfolio_id.in_(gids))

    df = pd.read_sql(s, db, index_col=['ra_portfolio_id', 'ra_key'])

    return df

# def max_id_between(min_id, max_id):
#     db = database.connection('asset')
#     metadata = MetaData(bind=db)
#     t = Table('ra_portfolio', metadata, autoload=True)

#     columns = [ t.c.globalid ]

#     s = select([func.max(t.c.globalid).label('maxid')]).where(t.c.globalid.between(min_id, max_id))

#     return s.execute().scalar()
def save(gid, df):
    #
    # 保存择时结果到数据库
    #
    db = database.connection('mapi')
    t2 = Table('fp_da_question', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns, (t2.c.fp_nare_id == gid))
    df_old = pd.read_sql(s, db, index_col=['globalid'])

    # 更新数据库
    print((df.head()))
    print("\n")
    print((df_old.head()))
    database.batch(db, t2, df.head(), df_old.head(), timestamp=True)

