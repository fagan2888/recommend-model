#coding=utf-8
'''
Created on: Mar. 7, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func, and_, or_
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_index_constituents(index_id, date=None, constituent_type='stock'):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_ix_comp', metadata, autoload=True)

    columns = [
        t.c.SELECTECODE.label(f'{constituent_type}_id'),
        t.c.SAMPLECODE.label(f'{constituent_type}_code')
    ]

    s = select(columns).where(t.c.SECODE==index_id)
    if date is None:
        s = s.where(t.c.OUTDATE=='19000101')
        # s = s.where(t.c.USESTATUS=='1')
    else:
        s = s.where(and_(t.c.SELECTEDDATE<=date, or_(t.c.OUTDATE>date, t.c.OUTDATE=='19000101')))

    df = pd.read_sql(s, engine, index_col=[f'{constituent_type}_id'])

    return df

def load_index_historical_constituents(index_id, begin_date=None, end_date=None, constituent_type='stock'):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_ix_comp', metadata, autoload=True)

    columns = [
        t.c.SECODE.label('index_id'),
        t.c.SELECTECODE.label(f'{constituent_type}_id'),
        t.c.SAMPLECODE.label(f'{constituent_type}_code'),
        t.c.SELECTEDDATE.label('selected_date'),
        t.c.OUTDATE.label('out_date')
    ]

    s = select(columns).where(t.c.SECODE==index_id)
    if begin_date is not None:
        s = s.where(or_(t.c.OUTDATE>begin_date, t.c.OUTDATE=='19000101'))
    if end_date is not None:
        s = s.where(t.c.SELECTEDDATE<=end_date)

    df = pd.read_sql(s, engine, parse_dates=['selected_date', 'out_date'])

    return df

if __name__ == '__main__':

    load_index_constiuents('2070000191')
    load_index_constiuents('2070000191', date='20181231')
    load_index_historical_constituents('2070000191')

