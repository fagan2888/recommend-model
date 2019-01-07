#coding=utf-8
'''
Created at Jan. 2, 2018
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_fund_inc_estimate(begin_date=None, end_date=None, fund_codes=None, methods=['sk_pos', 'ix_pos', 'mix'], to_update=False):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('fi_fund_inc_estimate', metadata, autoload=True)

    columns = [
            t.c.fi_trade_date,
            t.c.fi_fund_code,
            t.c.fi_inc_est_sk_pos,
            t.c.fi_inc_est_ix_pos,
            t.c.fi_inc_est_mix
    ]

    s = select(columns)
    if begin_date is not None:
        s = s.where(t.c.fi_trade_date>=begin_date)
    if end_date is not None:
        s = s.where(t.c.fi_trade_date<=end_date)
    if fund_codes is not None:
        s = s.where(t.c.fi_fund_code.in_(fund_codes))

    df = pd.read_sql(s, db, index_col=['fi_trade_date', 'fi_fund_code'], parse_dates=['fi_trade_date'])

    df = df[pd.Index(methods).map(lambda x: 'fi_inc_est_'+x)]

    if not to_update:
        df.index.names = ['date', 'fund_code']
        df.columns.name = 'method'
        df = df.rename(lambda x: x[11:], axis='columns')

    return df


def update_fund_inc_estimate(df_new, begin_date=None, end_date=None, fund_codes=None):

    df_new.loc[:] = round(df_new, 8)
    df_old = load_fund_inc_estimate(
            begin_date=begin_date,
            end_date=end_date,
            fund_codes=fund_codes,
            methods=df_new.columns,
            to_update=True
    )
    df_new.index.names = ['fi_trade_date', 'fi_fund_code']
    df_new = df_new.rename(lambda x: 'fi_inc_est_'+x, axis='columns')

    db = database.connection('asset')
    t = Table('fi_fund_inc_estimate', MetaData(bind=db), autoload=True)
    database.batch(db, t, df_new, df_old)


