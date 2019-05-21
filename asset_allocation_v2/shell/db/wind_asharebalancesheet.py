#coding=utf-8
'''
Created on: May. 19, 2019
Author: Ning Yang
Contact: yangning@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import multiprocessing
import numpy as np
import pandas as pd
from . import database
from . import util_db


logger = logging.getLogger(__name__)


def load_a_stock_balancesheet(stock_ids):

    stock_ids = util_db.to_list(stock_ids)

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    res = pool.map(load_a_stock_balancesheet_df, stock_ids)

    pool.close()
    pool.join()

    df = pd.concat(res)
    columns_float = df.columns.drop(['stock_id', 'ann_date', 'report_period'])
    df[columns_float] = df[columns_float].astype(float)

    return df

def load_a_stock_balancesheet_df(stock_id):

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareBalanceSheet', metadata, autoload=True)

    columns = [
        t.c.S_INFO_WINDCODE.label('stock_id'),
        t.c.ANN_DT.label('ann_date'),
        t.c.REPORT_PERIOD.label('report_period'),
        t.c.TOT_SHRHLDR_EQY_EXCL_MIN_INT.label('tot_shrhldr_euqity_excl_min_int'),
        t.c.TOT_ASSETS.label('tot_assets')
    ]

    s = select(columns).where(t.c.S_INFO_WINDCODE==stock_id).where(t.c.STATEMENT_TYPE=='408001000')
    df = pd.read_sql(s, engine, parse_dates=['ann_date', 'report_period'])

    return df

