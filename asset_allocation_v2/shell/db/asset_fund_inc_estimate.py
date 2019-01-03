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


def load_fund_inc_estimate(begin_date=None, end_date=None, fund_codes=None, methods=['sk_pos', 'ix_pos', 'mix']):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('fi_fund_inc_estimate', metadata, autoload=True)

    columns = [
            t.c.fi_trade_date.label('date'),
            t.c.fi_fund_code.label('fund_code'),
            t.c.fi_inc_estimate_sk_pos.label('sk_pos'),
            t.c.fi_inc_estimate_ix_pos.label('ix_pos'),
            t.c.fi_inc_estimate_mix.label('mix')
    ]

    s = select(columns)
    if begin_date is not None:
        s = s.where(t.c.fi_trade_date>=begin_date)
    if end_date is not None:
        s = s.where(t.c.fi_trade_date<=end_date)
    if fund_codes is not None:
        s = s.where(t.c.fi_fund_code.in_(fund_codes))

    df = pd.read_sql(s, db, index_col=['date', 'fund_code'], parse_dates=['date'])

    df = df[methods]

    return df


def update_fund_inc_estimate(df_new, last_date=None):

    df_old = load_fund_inc_estimate()
    db = database.connection('asset')
    t = Table('fi_fund_inc_estimate', MetaData(bind=db), autoload=True)
    database.batch(db, t, df_new, df_old)


