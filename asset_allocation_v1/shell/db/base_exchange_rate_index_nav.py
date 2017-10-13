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
# base.eri_index_nav
#
def load_series(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('exchange_rate_index_nav', metadata, autoload=True)

    columns = [
        t1.c.eri_date.label('date'),
        t1.c.eri_nav.label('nav'),
    ]

    s = select(columns).where(t1.c.eri_index_id == id_)
    
    if begin_date is not None:
        s = s.where(t1.c.eri_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.eri_date <= end_date)
    if mask is not None:
        if hasattr(mask, "__iter__") and not isinstance(mask, str):
            s = s.where(t1.c.eri_mask.in_(mask))
        else:
            s = s.where(t1.c.eri_mask == mask)
        
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['nav']

def load_ohlc(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('eri_index_nav', metadata, autoload=True)

    columns = [
        t1.c.eri_date,
        t1.c.eri_open,
        t1.c.eri_high,
        t1.c.eri_low,
        t1.c.eri_nav.label('eri_close'),
    ]

    s = select(columns).where(t1.c.eri_index_id == id_)
    
    if begin_date is not None:
        s = s.where(t1.c.eri_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.eri_date <= end_date)
    if mask is not None:
        if hasattr(mask, "__iter__") and not isinstance(mask, str):
            s = s.where(t1.c.eri_mask.in_(mask))
        else:
            s = s.where(t1.c.eri_mask == mask)
        
    df = pd.read_sql(s, db, index_col = ['eri_date'], parse_dates=['eri_date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df

def load_ohlcav(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('eri_index_nav', metadata, autoload=True)

    columns = [
        t1.c.eri_date,
        t1.c.eri_open,
        t1.c.eri_high,
        t1.c.eri_low,
        t1.c.eri_nav.label('eri_close'),
        t1.c.eri_amount,
        t1.c.eri_volume,
    ]

    s = select(columns).where(t1.c.eri_index_id == id_)

    if begin_date is not None:
        s = s.where(t1.c.eri_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.eri_date <= end_date)
    if mask is not None:
        if hasattr(mask, "__iter__") and not isinstance(mask, str):
            s = s.where(t1.c.eri_mask.in_(mask))
        else:
            s = s.where(t1.c.eri_mask == mask)

    df = pd.read_sql(s, db, index_col = ['eri_date'], parse_dates=['eri_date'])
    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df

