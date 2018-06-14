#coding=utf8

from sqlalchemy import MetaData, Table, select, func
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
# wt_filter_nav
#
def load_series(id_, filter_num, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)

    t = Table('wt_filter', metadata, autoload=True)
    columns = [
        t.c.globalid,
        t.c.wt_name,
        t.c.wt_filter_num,
        t.c.wt_index_id,
        t.c.wt_begin_date,
    ]
    s = select(columns)
    s = s.where(t.c.wt_index_id == id_)
    s = s.where(t.c.wt_filter_num == filter_num)
    df = pd.read_sql(s, db)

    t1 = Table('wt_filter_nav', metadata, autoload=True)

    columns = [
        t1.c.wt_date.label('date'),
        t1.c.wt_nav.label('nav'),
    ]

    s = select(columns).where(t1.c.wt_filter_id == df.iloc[0]['globalid'])

    if begin_date is not None:
        s = s.where(t1.c.wt_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.wt_date <= end_date)

    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['nav']
