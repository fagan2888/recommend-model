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
from . import util_db


logger = logging.getLogger(__name__)


def load_a_stock_free_float_share(stock_ids=None, begin_date=None, end_date=None):

    stock_ids = util_db.to_list(stock_ids)

    if begin_date is not None:
        begin_date = pd.Timestamp(begin_date).strftime('%Y%m%d')
    if end_date is not None:
        end_date = pd.Timestamp(end_date).strftime('%Y%m%d')

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareFreeFloat', metadata, autoload=True)

    columns = [
        t.c.S_INFO_WINDCODE.label('stock_id'),
        t.c.CHANGE_DT.label('trade_date'),
        # t.c.ANN_DT.label('ann_date'),
        t.c.S_SHARE_FREESHARES.label('free_float_share')
    ]

    s = select(columns)
    if stock_ids is not None:
        s = s.where(t.c.S_INFO_WINDCODE.in_(stock_ids))

    df = pd.read_sql(s, engine, parse_dates=['trade_date'])
    df.drop_duplicates(subset=['stock_id', 'trade_date'])
    df.set_index(['stock_id', 'trade_date'], inplace=True)
    df.sort_index(inplace=True)

    return df


if __name__ == '__main__':

    load_a_stock_free_float_share('601598.SH')
    load_a_stock_free_float_share(begin_date='2019-01-01', end_date='2019-04-30')

