'''
Created on: May. 7, 2019
Author: Ning Yang
Contact: yangning@licaimofang.com
'''
import logging
from sqlalchemy import MetaData, Table, select, func
import multiprocessing
import functools
import numpy as np
import pandas as pd
import sys
from . import database



logger = logging.getLogger(__name__)


def load_financial_statement_data(stock_ids, table_name, statement_columns_str):
    # table_name :str
    # statement_columns_str: str
    if isinstance(stock_ids, str):
        stock_ids = [stock_ids]
    elif isinstance(stock_ids, (tuple, set)):
        stock_ids = list(stock_ids)
    elif isinstance(stock_ids, dict):
        stock_ids = list(stock_ids.values())
    else:
        if isinstance(stock_ids, (pd.Index, pd.Series, pd.DataFrame)):
            stock_ids = stock_ids.values
        if isinstance(stock_ids, np.ndarray):
            stock_ids = stock_ids.reshape(-1).tolist()
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count // 2)
    kwargs = {'table_name': table_name, 'statement_columns_str': statement_columns_str}
    res = pool.map(functools.partial(load_financial_statement_data_ser, **kwargs), stock_ids)
    pool.close()
    pool.join()
    df = pd.concat(res)
    df.drop_duplicates(subset=['WIND_CODE', 'REPORT_PERIOD'], inplace=True)
    return df


def load_financial_statement_data_ser(stock_id, table_name=None, statement_columns_str=None):
    engine = database.connection('multi_factor')
    sql_t = 'select ' + statement_columns_str + ' from ' + table_name + ' where WIND_CODE = ' + str('"') + stock_id + str('"') + ' and ' + 'STATEMENT_TYPE = "408001000"'
    df = pd.read_sql(sql=sql_t, con=engine, parse_dates=['REPORT_PERIOD'])
    return df

# stock_ids = ['000001.SZ', '000002.SZ', '600001.SH']
# table_name = 'asharebalancesheet'
# statement_columns_str = 'WIND_CODE,ACTUAL_ANN_DT,REPORT_PERIOD,TOT_SHRHLDR_EQY_EXCL_MIN_INT,TOT_ASSETS'

if __name__ == '__main__':
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count // 2)
    kwargs = {'table_name': table_name, 'statement_columns': statement_columns_str}
    res = pool.map(functools.partial(load_financial_statement_data_ser, **kwargs), stock_ids)

