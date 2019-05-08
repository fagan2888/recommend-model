#coding=utf-8
'''
Created on: May. 8, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func, and_, or_
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_a_index_constituents(index_id, date=None):

    if date is not None:
        date = pd.Timestamp(date).strftime('%Y%m%d')

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AIndexMembers', metadata, autoload=True)

    columns = [
        t.c.S_CON_WINDCODE.label('stock_id')
    ]

    s = select(columns).where(t.c.S_INFO_WINDCODE==index_id)
    if date is None:
        s = s.where(t.c.S_CON_OUTDATE==None)
        # s = s.where(t.c.CUR_SIGN=='1')
    else:
        s = s.where(and_(t.c.S_CON_INDATE<=date, or_(t.c.S_CON_OUTDATE>=date, t.c.S_CON_OUTDATE==None)))

    df = pd.read_sql(s, engine, index_col=['stock_id'])
    df.sort_index(inplace=True)

    return df

def load_a_index_historical_constituents(index_id, begin_date=None, end_date=None):

    if begin_date is not None:
        begin_date = pd.Timestamp(begin_date).strftime('%Y%m%d')
    if end_date is not None:
        end_date = pd.Timestamp(end_date).strftime('%Y%m%d')

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AIndexMembers', metadata, autoload=True)

    columns = [
        t.c.S_INFO_WINDCODE.label('index_id'),
        t.c.S_CON_WINDCODE.label('stock_id'),
        t.c.S_CON_INDATE.label('in_date'),
        t.c.S_CON_OUTDATE.label('out_date')
    ]

    s = select(columns).where(t.c.S_INFO_WINDCODE==index_id)
    if begin_date is not None:
        s = s.where(or_(t.c.S_CON_OUTDATE>=begin_date, t.c.S_CON_OUTDATE==None))
    if end_date is not None:
        s = s.where(t.c.S_CON_INDATE<=end_date)

    df = pd.read_sql(s, engine, parse_dates=['in_date', 'out_date'])
    df.sort_values(['in_date', 'stock_id'], inplace=True)

    return df

if __name__ == '__main__':

    load_a_index_constiuents('000906.SH')
    load_a_index_constiuents('000906.SH', date='2018-12-31')
    load_a_index_historical_constituents('000906.SH')

