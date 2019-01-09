#coding=utf-8
'''
Created at Dec. 28, 2018
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_stock_nav(begin_date=None, end_date=None, reindex=None, stock_ids=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_sk_dquoteindic', metadata, autoload=True)

    columns = [
            t.c.TRADEDATE.label('date'),
            t.c.SECODE.label('stock_id'),
            t.c.TCLOSEAF.label('nav')
    ]

    s = select(columns)
    if begin_date is not None:
        s = s.where(t.c.TRADEDATE>=begin_date)
    if end_date is not None:
        s = s.where(t.c.TRADEDATE<=end_date)
    if stock_ids is not None:
        s = s.where(t.c.SECODE.in_(stock_ids))

    df = pd.read_sql(s, engine, parse_dates=['date'])

    df = df.pivot('date', 'stock_id', 'nav')
    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df


if __name__ == '__main__':

    load_stock_nav(begin_date='20181201', end_date='20181227')

