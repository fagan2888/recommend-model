#coding=utf-8
'''
Created on: May. 8, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_a_trade_date(exch_market_id='SSE', begin_date=None, end_date=None):

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareCalendar', metadata, autoload=True)

    columns = [
        t.c.TRADE_DAYS.label('trade_date'),
    ]

    s = select(columns).where(t.c.S_INFO_EXCHMARKET==exch_market_id)

    df = pd.read_sql(s, engine, index_col=['trade_date'], parse_dates=['trade_date'])
    ix = df.index
    ix.sort_values(inplace=True)

    if begin_date is not None:
        ix = ix[ix>=begin_date]
    if end_date is not None:
        ix = ix[ix<=end_date]

    return ix


if __name__ == '__main__':

    begin_date = pd.Timestamp('2000-01-01')
    end_date = pd.Timestamp('2018-12-31')
    load_a_trade_date(begin_date=begin_date, end_date=end_date)

