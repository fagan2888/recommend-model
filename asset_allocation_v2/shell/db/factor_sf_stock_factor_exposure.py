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
from . import database


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

