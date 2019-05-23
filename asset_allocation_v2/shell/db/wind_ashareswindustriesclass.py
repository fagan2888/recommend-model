#coding=utf-8
'''
Created on: May. 23, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func, and_, or_
import pandas as pd
from . import database
from . import util_db


logger = logging.getLogger(__name__)


def load_a_stock_sw_industry_level1(stock_ids=None, date=None):

    stock_ids = util_db.to_list(stock_ids)

    if date is not None:
        date = pd.Timestamp(date).strftime('%Y%m%d')

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareSWIndustriesClass', metadata, autoload=True)

    columns = [
        t.c.S_INFO_WINDCODE.label('stock_id'),
        t.c.SW_IND_CODE.label('sw_ind_code')
    ]

    s = select(columns)
    if stock_ids is not None:
        s = s.where(t.c.S_INFO_WINDCODE.in_(stock_ids))
    if date is None:
        s = s.where(t.c.REMOVE_DT==None)
        # s = s.where(t.c.CUR_SIGN=='1')
    else:
        s = s.where(and_(t.c.ENTRY_DT<=date, or_(t.c.REMOVE_DT>=date, t.c.REMOVE_DT==None)))

    df = pd.read_sql(s, engine, index_col=['stock_id'])
    df.sort_index(inplace=True)
    df['sw_ind_lv1_code'] = df.sw_ind_code.apply(lambda x: f'{x[:-6]}000000')

    return df

def load_a_stock_historical_sw_industry_level1(stock_ids):

    stock_ids = util_db.to_list(stock_ids)

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareSWIndustriesClass', metadata, autoload=True)

    columns = [
        t.c.S_INFO_WINDCODE.label('stock_id'),
        t.c.SW_IND_CODE.label('sw_ind_code'),
        t.c.ENTRY_DT.label('entry_date'),
        t.c.REMOVE_DT.label('remove_date')
    ]

    s = select(columns)
    if stock_ids is not None:
        s = s.where(t.c.S_INFO_WINDCODE.in_(stock_ids))

    df = pd.read_sql(s, engine, parse_dates=['entry_date', 'remove_date'])
    df.sort_values(by=['stock_id', 'entry_date'], inplace=True)
    df['sw_ind_lv1_code'] = df.sw_ind_code.apply(lambda x: f'{x[:-6]}000000')

    return df


if __name__ == '__main__':

    load_a_stock_sw_industry_level1('601318.SH')
    load_a_stock_sw_industry_level1('601318.SH', date='20131231')
    load_a_stock_sw_industry_level1('601318.SH', date='20140101')
    load_a_stock_historical_sw_industry_level1('601318.SH')

