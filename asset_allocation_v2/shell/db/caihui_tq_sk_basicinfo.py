#coding=utf-8
'''
Created on: Mar. 6, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func, tuple_
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_stock_code_info(stock_ids=None, stock_codes=None, current=True):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_sk_basicinfo', metadata, autoload=True)

    columns = [
        t.c.SECODE.label('stock_id'),
        t.c.SYMBOL.label('stock_code')
    ]

    s = select(columns)
    if stock_ids is not None:
        s = s.where(t.c.SECODE.in_(stock_ids))
    if stock_codes is not None:
        s = s.where(t.c.SYMBOL.in_(stock_codes))
    if current is True:
        columns2 = [
            t.c.SYMBOL,
            func.max(t.c.LISTDATE)
        ]
        s2 = select(columns2).group_by(t.c.SYMBOL)
        s = s.where(tuple_(t.c.SYMBOL, t.c.LISTDATE).in_(s2))

    df = pd.read_sql(s, engine, index_col=['stock_id'])

    return df

def load_stock_industry(stock_ids=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_sk_basicinfo', metadata, autoload=True)

    columns = [
        t.c.SECODE.label('stock_id'),
        t.c.CSRCLEVEL1CODE,
        t.c.CSRCLEVEL1NAME,
        t.c.CSRCLEVEL2CODE,
        t.c.CSRCLEVEL2NAME,
        t.c.GICSLEVEL1CODE,
        t.c.GICSLEVEL1NAME,
        t.c.GICSLEVEL2CODE,
        t.c.GICSLEVEL2NAME,
        t.c.SWLEVEL1CODE,
        t.c.SWLEVEL1NAME,
        t.c.SWLEVEL2CODE,
        t.c.SWLEVEL2NAME,
        t.c.CSILEVEL1CODE,
        t.c.CSILEVEL1NAME,
        t.c.CSILEVEL2CODE,
        t.c.CSILEVEL2NAME,
        t.c.FCLEVEL1CODE,
        t.c.FCLEVEL1NAME,
        t.c.FCLEVEL2CODE,
        t.c.FCLEVEL2NAME,
        t.c.PROVINCECODE,
        t.c.PROVINCENAME,
        t.c.CITYCODE,
        t.c.CITYNAME
    ]

    s = select(columns)
    if stock_ids is not None:
        s = s.where(t.c.SECODE.in_(stock_ids))

    df = pd.read_sql(s, engine, index_col=['stock_id'])

    return df

if __name__ == '__main__':

    load_stock_basic_info()

