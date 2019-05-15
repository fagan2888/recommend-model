#coding=utf-8
'''
Created on: May. 14, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database
from . import util_db

logger = logging.getLogger(__name__)


def load_a_stock_code_info(stock_ids=None, stock_codes=None, current=True):

    stock_ids = util_db.to_list(stock_ids)
    stock_codes = util_db.to_list(stock_codes)

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareDescription', metadata, autoload=True)

    columns = [
        t.c.S_INFO_WINDCODE.label('stock_id'),
        t.c.S_INFO_CODE.label('stock_code')
    ]

    s = select(columns)
    if stock_ids is not None:
        s = s.where(t.c.S_INFO_WINDCODE.in_(stock_ids))
    if stock_codes is not None:
        s = s.where(t.c.S_INFO_CODE.in_(stock_codes))
        if current is True:
            columns2 = [
                t.c.S_INFO_CODE,
                func.max(t.c.S_INFO_LISTDATE)
            ]
            s2 = select(columns2).group_by(t.c.S_INFO_CODE)
            s = s.where(tuple_(t.c.S_INFO_CODE, t.c.S_INFO_LISTDATE).in_(s2))

    df = pd.read_sql(s, engine, index_col=['stock_id'])
    df.sort_index(inplace=True)

    return df


if __name__ == '__main__':

    load_a_stock_code_info(stock_ids='601598.SH')

