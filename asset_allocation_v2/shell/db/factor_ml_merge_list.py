#coding=utf-8
'''
Created on: Apr. 08, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load():

    engine = database.connection('factor')
    metadata = MetaData(bind=engine)
    t = Table('ml_merge_list', metadata, autoload=True)

    columns = [
        t.c.trade_date,
        t.c.old_stock_id,
        t.c.new_stock_id,
        t.c.old_stock_price,
        t.c.new_stock_price,
        t.c.ratio
    ]

    s = select(columns)

    df = pd.read_sql(s, engine, parse_dates=['trade_date'])

    return df

