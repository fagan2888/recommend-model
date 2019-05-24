#coding=utf-8
'''
Created on: May. 22, 2019
Author: Ning Yang
Contact: yangning@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy.ext.declarative import declarative_base
import multiprocessing
import numpy as np
import pandas as pd
from . import database
from . import util_db
import pymysql


def query_table(table_name):
    engine = database.connection('multi_factor')
    Base = declarative_base()
    Base.metadata.reflect(engine)
    tables = Base.metadata.tables
    if table_name in tables.keys():
        metadata = MetaData(bind=engine)
        t = Table(table_name, metadata, autoload=True)
        columns = [t.c.trade_date.label('trade_date')]
        s = select(columns)
        df_tradedate = pd.read_sql(s, engine, parse_dates=['trade_date'])
        existing_tradedate = list(df_tradedate.trade_date.unique())
    else:
        existing_tradedate = list()
    return existing_tradedate


def write_factor(df_descriptor, table_name):
    df_descriptor_t = df_descriptor.reset_index().copy()
    engine = database.connection('multi_factor')
    for i in range(int(df_descriptor_t.shape[0] / 3000) + 1):
        df_descriptor_t2 = df_descriptor_t.iloc[3000 * i:3000 * (i + 1)]
        pd.io.sql.to_sql(df_descriptor_t2, table_name, con=engine, if_exists='append', index=False)
    return 'finished'

