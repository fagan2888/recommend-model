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
 
 
def load_a_stock_income(stock_ids):
 
    stock_ids = util_db.to_list(stock_ids)
 
    
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)
 
    res = pool.map(load_a_stock_income_df, stock_ids)
 
    pool.close()
    pool.join()
 
    df = pd.concat(res)
    columns_float = df.columns.drop(['stock_id', 'report_period'])
    df[columns_float] = df[columns_float].astype(float)
    return df
 
 
def load_a_stock_income_df(stock_id):
 
    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareIncome', metadata, autoload=True)
 
    columns = [
        t.c.S_INFO_WINDCODE.label('stock_id'),
        t.c.REPORT_PERIOD.label('report_period'),
        t.c.NET_PROFIT_AFTER_DED_NR_LP.label('net_profit_after_ded_nr_lp'),
        t.c.OPER_REV.label('oper_rev'),
        t.c.LESS_OPER_COST.label('less_oper_cost'),
    ]

    s = select(columns).where(t.c.S_INFO_WINDCODE  == stock_id).where(t.c.STATEMENT_TYPE == '408001000')
    df = pd.read_sql(s, engine, parse_dates=['report_period'])

    return df

