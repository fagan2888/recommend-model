#coding=utf-8
'''
Created on: Dec. 28, 2018
Modified on: Mar. 6, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_stock_price(begin_date=None, end_date=None, reindex=None, stock_ids=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_sk_dquoteindic', metadata, autoload=True)

    columns = [
        t.c.TRADEDATE.label('trade_date'),
        t.c.SECODE.label('stock_id'),
        t.c.TCLOSEAF.label('prc')
    ]

    s = select(columns)
    if begin_date is not None:
        s = s.where(t.c.TRADEDATE>=begin_date)
    if end_date is not None:
        s = s.where(t.c.TRADEDATE<=end_date)
    if stock_ids is not None:
        s = s.where(t.c.SECODE.in_(stock_ids))

    df = pd.read_sql(s, engine, parse_dates=['trade_date'])

    df = df.pivot('trade_date', 'stock_id', 'prc')
    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df

def load_stock_market_data(begin_date=None, end_date=None, stock_ids=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_sk_dquoteindic', metadata, autoload=True)

    columns = [
        t.c.SYMBOL.label('stock_code'),
        t.c.SECODE.label('stock_id'),
        t.c.TRADEDATE.label('trade_date'),
        t.c.TCLOSEAF.label('price'),
        t.c.VOL.label('volume'),
        t.c.AMOUNT.label('amount'),
        t.c.MKTSHARE.label('market_share'),
        t.c.TOTALSHARE.label('total_share')
    ]

    s = select(columns)
    if begin_date is not None:
        s = s.where(t.c.TRADEDATE>=begin_date)
    if end_date is not None:
        s = s.where(t.c.TRADEDATE<=end_date)
    if stock_ids is not None:
        s = s.where(t.c.SECODE.in_(stock_ids))

    df = pd.read_sql(s, engine, parse_dates=['trade_date'])

    df = df.set_index(['stock_id', 'trade_date'])

    return df

if __name__ == '__main__':

    load_stock_nav(begin_date='20181201', end_date='20181227')
    load_stock_dquoteindic(begin_date='20000101')


