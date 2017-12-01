#coding=utf8


from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database
import MySQLdb
import config

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# base.ra_index_nav
#
def closeaf(globalid = None, reindex=None, begin_date=None, end_date=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_stock_nav', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.sk_tradedate.label('date'),
        t1.c.sk_closeaf.label('nav'),
    ]

    s = select(columns)

    if globalid is not None:
        s = s.where(t1.c.globalid == globalid)
    if begin_date is not None:
        s = s.where(t1.c.sk_tradedate >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.sk_tradedate <= end_date)

    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df


'''
def load_ohlc(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_index_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date,
        t1.c.ra_open,
        t1.c.ra_high,
        t1.c.ra_low,
        t1.c.ra_nav.label('ra_close'),
    ]

    s = select(columns).where(t1.c.ra_index_id == id_)

    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
    if mask is not None:
        if hasattr(mask, "__iter__") and not isinstance(mask, str):
            s = s.where(t1.c.ra_mask.in_(mask))
        else:
            s = s.where(t1.c.ra_mask == mask)

    df = pd.read_sql(s, db, index_col = ['ra_date'], parse_dates=['ra_date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df

def load_ohlcav(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_index_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date,
        t1.c.ra_open,
        t1.c.ra_high,
        t1.c.ra_low,
        t1.c.ra_nav.label('ra_close'),
        t1.c.ra_amount,
        t1.c.ra_volume,
    ]

    s = select(columns).where(t1.c.ra_index_id == id_)

    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
    if mask is not None:
        if hasattr(mask, "__iter__") and not isinstance(mask, str):
            s = s.where(t1.c.ra_mask.in_(mask))
        else:
            s = s.where(t1.c.ra_mask == mask)

    df = pd.read_sql(s, db, index_col = ['ra_date'], parse_dates=['ra_date'])
    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df
'''
