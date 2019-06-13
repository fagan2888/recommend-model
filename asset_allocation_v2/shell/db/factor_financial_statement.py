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
from . import database
from . import util_db


logger = logging.getLogger(__name__)


def load_financial_statement_data(stock_ids, table_name, statement_columns_str):

    stock_ids = util_db.to_list(stock_ids)

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count // 2)

    kwargs = {'table_name': table_name, 'statement_columns_str': statement_columns_str}
    res = pool.map(functools.partial(load_financial_statement_data_ser, **kwargs), stock_ids)

    pool.close()
    pool.join()

    df = pd.concat(res)
    df.drop_duplicates(subset=['WIND_CODE', 'REPORT_PERIOD'], inplace=True)

    return df

def load_financial_statement_data_ser(stock_id, table_name, statement_columns_str=None):

    if statement_columns_str is None:
        statement_columns_str = '*'

    engine = database.connection('factor')

    sql = f'SELECT {statement_columns_str} FROM {table_name} WHERE WIND_CODE = \'{stock_id}\' AND STATEMENT_TYPE = \'408001000\''
    df = pd.read_sql(sql, con=engine, parse_dates=['REPORT_PERIOD'])

    return df


if __name__ == '__main__':

    stock_ids = ['000001.SZ', '000002.SZ', '600001.SH']
    table_name = 'asharebalancesheet'
    statement_columns_str = 'WIND_CODE,ACTUAL_ANN_DT,REPORT_PERIOD,TOT_SHRHLDR_EQY_EXCL_MIN_INT,TOT_ASSETS'

    load_financial_statement_data(stock_ids, table_name, statement_columns_str)

