#coding=utf8

from sqlalchemy import MetaData, Table, select, func
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
# base.ra_index_nav
#
def load_series(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_index_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date.label('date'),
        t1.c.ra_nav.label('nav'),
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
        
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['nav']

def load_onemore_week(_id, date):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_index_nav', metadata, autoload=True)
    t2 = Table('trade_dates', metadata, autoload=True)

    columns = [
        t1.c.ra_date.label('date'),
        t1.c.ra_nav,
    ]

    s = select(columns) \
        .select_from(t1.join(t2, t1.c.ra_date == t2.c.td_date)) \
        .where(t1.c.ra_date >= date) \
        .where((t2.c.td_date >= date) & (t2.c.td_type.op('&')(0x02))).limit(2).order_by(t1.c.ra_date)

    s = s.where(t1.c.ra_index_id == _id)
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    df = df.pct_change().fillna(0.0)
    if df.empty:
        return 0.0

    return df.iloc[-1].values[0]


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

