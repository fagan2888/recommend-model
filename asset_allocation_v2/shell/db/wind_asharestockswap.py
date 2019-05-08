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


def load_a_stock_swap(begin_date=None, end_date=None):

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareStockSwap', metadata, autoload=True)

    columns = [
        t.c.TRANSFERER_WINDCODE.label('transferer_stock_id'),
        t.c.TARGETCOMP_WINDCODE.label('targetcomp_stock_id'),
        t.c.TRANSFERER_CONVERSIONPRICE.label('transferer_stock_price'),
        t.c.TARGETCOMP_CONVERSIONPRICE.label('targetcomp_stock_price'),
        t.c.CONVERSIONRATIO.label('conversion_ratio'),
        t.c.EQUITYREGISTRATIONDATE.label('trade_date')
    ]

    s = select(columns)

    df = pd.read_sql(s, engine, parse_dates=['trade_date'])
    df.sort_values(['trade_date', 'transferer_stock_id'], inplace=True)

    return df

