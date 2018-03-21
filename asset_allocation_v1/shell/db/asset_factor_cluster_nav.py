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

def load_series(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('factor_cluster_nav', metadata, autoload=True)

    columns = [
        t1.c.date,
        t1.c.nav,
    ]

    s = select(columns).where(t1.c.fc_cluster_id == id_)

    if begin_date is not None:
        s = s.where(t1.c.date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.date <= end_date)

    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['nav']


def load_all(globalid):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('factor_cluster_nav', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.fc_cluster_id,
        t1.c.date,
        t1.c.nav,
    ]

    s = select(columns).where(t1.c.globalid == globalid)

    df = pd.read_sql(s, db)

    return df
