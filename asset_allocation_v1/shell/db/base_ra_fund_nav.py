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
# base.ra_fund_nav
#
def load_weekly(begin_date, end_date, fund_ids=None, codes=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_fund_nav', metadata, autoload=True)
    t2 = Table('trade_dates', metadata, autoload=True)

    columns = [
        t1.c.ra_code.label('code'),
        t1.c.ra_date.label('date'),
        t1.c.ra_nav_adjusted,
    ]

    s = select(columns) \
        .select_from(t1.join(t2, t1.c.ra_date == t2.c.td_date)) \
        .where(t1.c.ra_date.between(begin_date, end_date)) \
        .where(t2.c.td_date.between(begin_date, end_date) & (t2.c.td_type.op('&')(0x02) | (t2.c.td_date == end_date)))
    
    if fund_ids is not None:
        s = s.where(t1.c.ra_fund_id.in_(fund_ids))

    if codes is not None:
        s = s.where(t1.c.ra_code.in_(codes))

    df = pd.read_sql(s, db, index_col = ['date', 'code'], parse_dates=['date'])

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df

def load_daily(begin_date, end_date, reindex=None, fund_ids=None, codes=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_fund_nav', metadata, autoload=True)
    t2 = Table('trade_dates', metadata, autoload=True)

    columns = [
        t1.c.ra_code.label('code'),
        t1.c.ra_date.label('date'),
        t1.c.ra_nav_adjusted,
    ]

    s = select(columns) \
        .select_from(t1.join(t2, t1.c.ra_date == t2.c.td_date)) \
        .where(t1.c.ra_date.between(begin_date, end_date)) \
        .where(t2.c.td_date.between(begin_date, end_date))
    
    if fund_ids is not None:
        s = s.where(t1.c.ra_fund_id.in_(fund_ids))

    if codes is not None:
        s = s.where(t1.c.ra_code.in_(codes))

    df = pd.read_sql(s, db, index_col = ['date', 'code'], parse_dates=['date'])

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df

def load_series(code, reindex=None, begin_date=None, end_date=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_fund_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date.label('date'),
        t1.c.ra_nav_adjusted.label('nav'),
    ]

    s = select(columns) \
        .where((t1.c.ra_code == code) | (t1.c.ra_fund_id == code))

    # s = select(columns).where(t1.c.ra_fund_id == id_)
    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['nav']


    
