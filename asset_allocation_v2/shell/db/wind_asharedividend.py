#coding=utf-8
'''
Created on: Jun. 3, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database
from . import util_db


logger = logging.getLogger(__name__)


def load_a_stock_dividend(stock_ids=None):

    stock_ids = util_db.to_list(stock_ids)

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareDividend', metadata, autoload=True)

    columns = [
        t.c.S_INFO_WINDCODE.label('stock_id'),
        t.c.EQY_RECORD_DT.label('eqy_record_date'),
        t.c.EX_DT.label('ex_date'),
        t.c.DVD_ANN_DT.label('dvd_ann_date'),
        t.c.ANN_DT.label('ann_date'),
        t.c.STK_DVD_PER_SH.label('stock_dvd_per_sh'),
        t.c.CASH_DVD_PER_SH_PRE_TAX.label('cash_dvd_per_sh_pre_tax'),
        t.c.CASH_DVD_PER_SH_AFTER_TAX.label('cash_dvd_per_sh_after_tax'),
        t.c.S_DIV_BONUSRATE.label('div_bonus_rate'),
        t.c.S_DIV_CONVERSEDRATE.label('div_conversed_rate'),
        t.c.S_DIV_PROGRESS.label('div_progress'),
        t.c.IS_TRANSFER.label('is_not_transfer')
    ]

    s = select(columns)
    if stock_ids is not None:
        s = s.where(t.c.S_INFO_WINDCODE.in_(stock_ids))

    df = pd.read_sql(s, engine, parse_dates=['eqy_record_date', 'ex_date', 'dvd_ann_date', 'ann_date'])
    df.sort_values(by=['stock_id', 'eqy_record_date'], inplace=True)

    return df


if __name__ == '__main__':

    load_a_stock_dividend('600036.SH')

