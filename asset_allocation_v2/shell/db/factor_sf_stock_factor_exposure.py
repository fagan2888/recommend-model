#coding=utf-8
'''
Created on: May. 22, 2019
Author: Ning Yang
Contact: yangning@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func, literal_column
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import multiprocessing
from . import database
from . import util_db
from functools import partial


logger = logging.getLogger(__name__)


def query_table(table_name):

    engine = database.connection('factor')
    Base = declarative_base()
    Base.metadata.reflect(engine)
    tables = Base.metadata.tables
    if table_name in tables.keys():
        metadata = MetaData(bind=engine)
        t = Table(table_name, metadata, autoload=True)
        columns = [t.c.trade_date]
        s = select(columns)
        df_tradedate = pd.read_sql(s, engine, parse_dates=['trade_date'])
        existing_tradedate = df_tradedate.trade_date.unique()
    else:
        existing_tradedate = pd.Index()

    return existing_tradedate


def write_factor(df, table_name):

    db = database.connection('factor')
    t2 = Table(table_name, MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    df_old = pd.DataFrame(columns=list(df.index.names)+list(df.columns)).set_index(['trade_date', 'stock_id'])
    database.batch(db, t2, df, df_old, timestamp=False)



def load_a_stock_factor_exposure(stock_ids, style_factor):   
    stock_ids = util_db.to_list(stock_ids)
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)
    res = pool.map(partial(load_a_stock_factor_exposure_df, style_factor=style_factor), stock_ids)
    pool.close()
    pool.join()
    df = pd.concat(res)
    df = df.set_index(['trade_date', 'stock_id']).sort_index()

    return df
 
def load_a_stock_factor_exposure_df(stock_id, style_factor):  
    engine = database.connection('factor')
    column_str = str()
    for i_style in style_factor:
        column_str = column_str + i_style + ', '
    column_str = column_str + 'trade_date, stock_id, weight, sw_lv1_ind_code'

    sql_t = 'select ' + column_str + ' from sf_stock_factor_exposure_z_score where stock_id=' + '"' + stock_id + '"'   
    df = pd.read_sql(sql_t, engine, parse_dates=['trade_date'])
  
    return df


def load_factor_return(style_factor):
    engine = database.connection('factor')
    column_str = str()
    for i_style in style_factor:
        column_str = column_str + i_style + ', '
    column_str = column_str + 'trade_date'
    sql_t = 'select ' + column_str + ' from fr_factor_return_half_month'
    df = pd.read_sql(sql_t, engine, parse_dates=['trade_date'], index_col=['trade_date']).sort_index()
    
    return df

