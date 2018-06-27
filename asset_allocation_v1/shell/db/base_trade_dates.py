#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database
#  import MySQLdb
import config
import re

from dateutil.parser import parse

logger = logging.getLogger(__name__)


#
# base.trade_dates
#
def load_index(begin_date=None, end_date=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('trade_dates', metadata, autoload=True)

    columns = [
        t1.c.td_date,
        t1.c.td_type,
    ]

    s = select(columns)
    if begin_date is not None:
        s = s.where(t1.c.td_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.td_date <= end_date)

    df = pd.read_sql(s, db, index_col = ['td_date'], parse_dates=['td_date'])

    return df.index


def load_origin_index_trade_date(index_id, begin_date=None, end_date=None):

    if index_id.isdigit():
        xtype = int(index_id) / 10000000
    else:
        xtype = re.sub(r'([\d]+)','',index_id).strip()

    db = database.connection('base')
    metadata = MetaData(bind=db)
    if xtype == 'ERI':
        t1 = Table('exchange_rate_index_nav', metadata, autoload=True)

        columns = [
            t1.c.eri_nav_date,
        ]

        s = select(columns).where(t1.c.eri_index_id == index_id)
        if begin_date is not None:
            s = s.where(t1.c.eri_nav_date >= begin_date)
        if end_date is not None:
            s = s.where(t1.c.eri_nav_date <= end_date)

        df = pd.read_sql(s, db, index_col = ['eri_nav_date'], parse_dates=['eri_nav_date'])

    elif xtype == 12:
        t1 = Table('ra_index_nav', metadata, autoload=True)

        columns = [
            t1.c.ra_nav_date,
        ]

        s = select(columns).where(t1.c.ra_index_id == index_id)
        if begin_date is not None:
            s = s.where(t1.c.ra_nav_date >= begin_date)
        if end_date is not None:
            s = s.where(t1.c.ra_nav_date <= end_date)

        df = pd.read_sql(s, db, index_col = ['ra_nav_date'], parse_dates=['ra_nav_date'])

    else:
        return None

    df = df.groupby(df.index).first()

    return df.index


def load_trade_dates(begin_date=None, end_date=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('trade_dates', metadata, autoload=True)

    columns = [
        t1.c.td_date,
        t1.c.td_type,
    ]

    s = select(columns)
    if begin_date is not None:
        s = s.where(t1.c.td_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.td_date <= end_date)

    df = pd.read_sql(s, db, index_col = ['td_date'], parse_dates=['td_date'])

    return df

def trade_date_lookback_index(end_date=None, lookback=26, include_end_date=True):
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('trade_dates', metadata, autoload=True)
    columns = [
        t1.c.td_date.label('date'),
        t1.c.td_type,
    ]

    s = select(columns).where(t1.c.td_date <= end_date)

    if include_end_date:
        s = s.where((t2.c.td_type.op('&')(0x02)) | (t1.c.td_date == end_date))
    else:
        s = s.where((t1.c.td_type.op('&')(0x02)))

    s = s.order_by(t2.c.td_date.desc()).limit(lookback)

    df = pd.read_sql(s, db, index_col = 'date', parse_dates=True)

    return df.index.sort_values()

#  def trade_date_lookback_index(end_date=None, lookback=26, include_end_date=True):
    #  if include_end_date:
        #  condition = "(td_type & 0x02 OR td_date = '%s')" % (end_date)
    #  else:
        #  condition = "(td_type & 0x02)"

    #  sql = "SELECT td_date as date, td_type FROM trade_dates WHERE td_date <= '%s' AND %s ORDER By td_date DESC LIMIT %d" % (end_date, condition, lookback)

    #  conn  = MySQLdb.connect(**config.db_base)
    #  df = pd.read_sql(sql, conn, index_col = 'date', parse_dates=['date'])
    #  conn.close()

    #  return df.index.sort_values()
