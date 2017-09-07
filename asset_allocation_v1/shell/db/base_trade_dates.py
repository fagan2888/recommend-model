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


