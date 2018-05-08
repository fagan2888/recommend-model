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

def load_series(gid, risk):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('mz_highlow_limit', metadata, autoload=True)

    columns = [
        t1.c.mz_date,
        t1.c.ratio_h,
        t1.c.ratio_l,
    ]

    s = select(columns).where(t1.c.mz_highlow_id == gid).where(t1.c.mz_risk == risk)

    df = pd.read_sql(s, db, index_col = ['mz_date'], parse_dates=['mz_date'])

    return df


def save(gid, risk, df):
    # fmt_columns = ['mz_nav', 'mz_inc']
    # fmt_precision = 6
    # if not df.empty:
    #     df = database.number_format(df, fmt_columns, fmt_precision)
    #
    # 保存择时结果到数据库
    #
    db = database.connection('asset')
    t2 = Table('mz_highlow_limit', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns, (t2.c.mz_highlow_id == gid)).where(t2.c.mz_risk == risk)
    df_old = pd.read_sql(s, db, index_col=['globalid', 'mz_highlow_id', 'mz_risk', 'mz_date'], parse_dates=['mz_date'])
    # if not df_old.empty:
    #     df_old = database.number_format(df_old, fmt_columns, fmt_precision)

    # 更新数据库
    # print df_new.head()
    # print df_old.head()
    database.batch(db, t2, df, df_old, timestamp=False)
