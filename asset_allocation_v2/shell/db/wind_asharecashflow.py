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


def load_a_stock_cashflow(stock_ids):

    stock_ids = util_db.to_list(stock_ids)

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    res = pool.map(load_a_stock_cashflow_df, stock_ids)

    pool.close()
    pool.join()

    df = pd.concat(res)
    columns_float = df.columns.drop(['stock_id', 'report_period'])
    df[columns_float] = df[columns_float].astype(float)

    return df

def load_a_stock_cashflow_df(stock_id):

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareCashFlow', metadata, autoload=True)

    columns = [
        t.c.S_INFO_WINDCODE.label('stock_id'),
        t.c.REPORT_PERIOD.label('report_period'),
        t.c.DEPR_FA_COGA_DPBA.label('depr_fa_coga_dpba'),
        t.c.AMORT_INTANG_ASSETS.label('amort_intang_assets'),
        t.c.AMORT_LT_DEFERRED_EXP.label('amort_lt_deferred_exp'),
        t.c.NET_CASH_FLOWS_OPER_ACT.label('net_cash_flows_oper_act'),
        t.c.DECR_INVENTORIES.label('decr_inventories'),
        t.c.DECR_OPER_PAYABLE.label('decr_oper_payable'),
        t.c.INCR_OPER_PAYABLE.label('incr_oper_payable'),
        t.c.OTHERS.label('others')
    ]

    s = select(columns).where(t.c.S_INFO_WINDCODE==stock_id).where(t.c.STATEMENT_TYPE=='408001000')
    df = pd.read_sql(s, engine, parse_dates=['report_period'])

    return df

