#coding=utf-8
'''
Created on: Mar. 13, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func, or_, tuple_
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_company_share(company_ids=None, date=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_sk_sharestruchg', metadata, autoload=True)

    columns = [
        t.c.COMPCODE.label('company_id'),
        t.c.TOTALSHARE.label('total_share'),
        t.c.FCIRCAAMT.label('free_float_share')
    ]

    s = select(columns)
    if company_ids is not None:
        s = s.where(t.c.COMPCODE.in_(company_ids))
    if date is None:
        columns2 = [
            t.c.COMPCODE,
            func.MAX(t.c.BEGINDATE)
        ]
        s2 = select(columns2).group_by(t.c.COMPCODE)
        s = s.where(tuple_(t.c.COMPCODE, t.c.BEGINDATE).in_(s2))
    else:
        s = s.where(t.c.BEGINDATE<=date)
        s = s.where(or_(t.c.ENDDATE>date, t.c.ENDDATE=='19000101'))

    df = pd.read_sql(s, engine, index_col=['company_id'])

    return df

def load_company_historical_share(company_ids=None, begin_date=None, end_date=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_sk_sharestruchg', metadata, autoload=True)

    columns = [
        t.c.COMPCODE.label('company_id'),
        t.c.BEGINDATE.label('begin_date'),
        t.c.ENDDATE.label('end_date'),
        t.c.TOTALSHARE.label('total_share'),
        t.c.FCIRCAAMT.label('free_float_share')
    ]

    s = select(columns)
    if company_ids is not None:
        s = s.where(t.c.COMPCODE.in_(company_ids))
    if begin_date is not None:
        s = s.where(or_(t.c.ENDDATE>begin_date, t.c.ENDDATE=='19000101'))
    if end_date is not None:
        s = s.where(t.c.BEGINDATE<=end_date)

    df = pd.read_sql(s, engine, parse_dates=['begin_date', 'end_date'])

    return df

if __name__ == '__main__':

    load_company_share(company_ids=['10000001', '10000013', '10000020'])
    load_company_share(date='20150101')
    load_company_historical_share(begin_date='20160101', end_date='20170101')

