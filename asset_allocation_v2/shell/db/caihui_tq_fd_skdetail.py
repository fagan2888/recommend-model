#coding=utf-8
'''
Created at Dec. 28, 2018
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func, and_
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_fund_skdetail_all(end_date, fund_ids=None):

    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table('tq_fd_skdetail', metadata, autoload=True)

    columns = [
            t.c.SECODE.label('fund_id'),
            t.c.SKCODE.label('stock_id'),
            t.c.NAVRTO.label('navrto')
    ]

    s = select(columns)
    s = s.where(t.c.ENDDATE==end_date)
    if fund_ids is not None:
        s = s.where(t.c.SECODE.in_(fund_ids))

    df = pd.read_sql(s, db, index_col=['fund_id', 'stock_id'])

    return df


def load_fund_skdetail_ten(end_date, publish_date, fund_ids=None):

    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table('tq_fd_skdetail', metadata, autoload=True)

    columns = [
            t.c.SECODE.label('fund_id'),
            t.c.SKCODE.label('stock_id'),
            t.c.NAVRTO.label('navrto')
    ]

    s = select(columns)
    s = s.where(and_(
            t.c.ENDDATE==end_date,
            t.c.PUBLISHDATE<publish_date
    ))
    if fund_ids is not None:
        s = s.where(t.c.SECODE.in_(fund_ids))

    df = pd.read_sql(s, db, index_col=['fund_id', 'stock_id'])

    return df


if __name__ == '__main__':
    load_fd_skdetail_all('20180630')
    load_fd_skdetail_ten('20180630', '20180801')

