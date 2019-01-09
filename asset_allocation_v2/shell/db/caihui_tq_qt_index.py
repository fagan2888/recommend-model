#coding=utf-8
'''
Edited at Dec. 28, 2018
Editor: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_index_daily_data(secode, start_date=None, end_date=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_qt_index', metadata, autoload=True)

    columns = [
        t.c.TRADEDATE.label('date'),
        t.c.TCLOSE.label('close'),
        t.c.THIGH.label('high'),
        t.c.TLOW.label('low'),
        t.c.VOL.label('volume'),
        t.c.TOPEN.label('open'),
    ]

    s = select(columns).where(t.c.SECODE == secode)
    if start_date:
        s = s.where(t.c.TRADEDATE >= start_date)
    if end_date:
        s = s.where(t.c.TRADEDATE <= end_date)
    s = s.where(t.c.ISVALID == 1).order_by(t.c.TRADEDATE.asc())

    df = pd.read_sql(s, engine, index_col = ['date'], parse_dates=['date'])

    return df


def load_index_nav(begin_date=None, end_date=None, reindex=None, index_ids=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_qt_index', metadata, autoload=True)

    columns = [
            t.c.TRADEDATE.label('date'),
            t.c.SECODE.label('index_id'),
            t.c.TCLOSE.label('nav')
    ]

    s = select(columns)
    if begin_date is not None:
        s = s.where(t.c.TRADEDATE>=begin_date)
    if end_date is not None:
        s = s.where(t.c.TRADEDATE<=end_date)
    if index_ids is not None:
        s = s.where(t.c.SECODE.in_(index_ids))

    df = pd.read_sql(s, engine, parse_dates=['date'])

    df = df.pivot('date', 'index_id', 'nav')
    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df


if __name__ == '__main__':

    load_index_daily_data('2070006540', '20170101', '20170331')
    load_index_nav(index_ids=['2070000005', '2070000014', '2070000553', '2070000060', '2070000187'])
