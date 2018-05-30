#coding=utf8

from sqlalchemy import MetaData, Table, select, func
import pandas as pd
import logging
import database
from dateutil.parser import parse
import sys
from ipdb import set_trace
logger = logging.getLogger(__name__)
def load_index_daily_data(secode, start_date=None, end_date=None):
    db = database.connection('caihui')
    metadata = MetaData(bind=db)
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
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])
    return df


def load_multi_index_nav(secodes, start_date=None, end_date=None):

    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table('tq_qt_index', metadata, autoload=True)

    columns = [
        t.c.TRADEDATE.label('date'),
        t.c.SECODE.label('secode'),
        t.c.TCLOSE.label('close'),
    ]
    s = select(columns).where(t.c.SECODE.in_(secodes))

    if start_date:
        s = s.where(t.c.TRADEDATE >= start_date)
    if end_date:
        s = s.where(t.c.TRADEDATE <= end_date)

    s = s.where(t.c.ISVALID == 1)

    df = pd.read_sql(s, db, index_col = ['date', 'secode'], parse_dates=['date'])
    df = df.unstack().fillna(method = 'pad')
    df.columns = df.columns.get_level_values(1)

    return df


if __name__ == "__main__":
    # load_index_daily_data('2070006540', '20170101', '20170331')
    # df = load_multi_index_nav(['2070000176', '2070006540'])
    df = load_multi_index_nav(['2070006540', '2070004620'])
    print df.head()
    print df.tail()








