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
        t.c.CSRCLEVEL1CODE.label('csrc_level1_code'),
        t.c.CSRCLEVEL1NAME.label('csrc_level1_name'),
        t.c.CSRCLEVEL2CODE.label('csrc_level2_code'),
        t.c.CSRCLEVEL2NAME.label('csrc_level2_name'),
        t.c.GICSLEVEL1CODE.label('gics_level1_code'),
        t.c.GICSLEVEL1NAME.label('gics_level1_name'),
        t.c.GICSLEVEL2CODE.label('gics_level2_code'),
        t.c.GICSLEVEL2NAME.label('gics_level2_name'),
        t.c.SWLEVEL1CODE.label('sw_level1_code'),
        t.c.SWLEVEL1NAME.label('sw_level1_name'),
        t.c.SWLEVEL2CODE.label('sw_level2_code'),
        t.c.SWLEVEL2NAME.label('sw_level2_name'),
        t.c.CSILEVEL1CODE.label('csi_level1_code'),
        t.c.CSILEVEL1NAME.label('csi_level1_name'),
        t.c.CSILEVEL2CODE.label('csi_level2_code'),
        t.c.CSILEVEL2NAME.label('csi_level2_name'),
        t.c.FCLEVEL1CODE.label('fc_level1_code'),
        t.c.FCLEVEL1NAME.label('fc_level1_name'),
        t.c.FCLEVEL2CODE.label('fc_level2_code'),
        t.c.FCLEVEL2NAME.label('fc_level2_name'),
        t.c.PROVINCECODE.label('province_code'),
        t.c.PROVINCENAME.label('province_name'),
        t.c.CITYCODE.label('city_code'),
        t.c.CITYNAME.label('city_name')
    ]

    s = select(columns)
    if stock_ids is not None:
        s = s.where(t.c.SECODE.in_(stock_ids))

    df = pd.read_sql(s, engine, index_col=['stock_id'])

    return df

if __name__ == '__main__':

    load_stock_basic_info()

