#coding=utf-8
'''
Created at Dec. 28, 2018
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import sys
import logging
sys.path.append('shell')
sys.path.append('..')
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from dateutil.parser import parse
from . import database


logger = logging.getLogger(__name__)


def load_fd_skdetail_all(fund_ids=None, end_date=None):

    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table('tq_fd_basicinfo', metadata, autoload=True)

    columns = [
            t.c.SECODE.label('fund_id'),
            t.c.SKCODE.label('stock_id'),
            t.c.NAVRTO.label('navrto')
    ]

    s = select(columns)
    # if end_date is None:
        # end_date = last_end_date_fund_skdetail_all_published(pd.Timestamp.now())
    s = s.where(t.c.ENDDATE==end_date)
    if fund_ids is not None:
        s = s.where(t.c.SECODE.in_(fund_ids))

    df = pd.read_sql(s, db, index_col=['fund_id', 'stock_id'])

    return df


def load_fd_skdetail_ten(fund_ids=None, end_date=None):

    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table('tq_fd_basicinfo', metadata, autoload=True)

    columns = [
            t.c.SECODE.label('fund_id'),
            t.c.SKCODE.label('stock_id'),
            t.c.NAVRTO.label('navrto')
    ]

    s = select(columns)
    # if end_date is None:
        # end_date = last_end_date_fund_skdetail_ten_published(pd.Timestamp.now())
    # publish_date = next_month(date_as_timestamp(end_date))
    s = s.where(
            t.c.ENDDATE==end_date
            # t.c.PUBLISHDATE<=date_as_str(publish_date)
    )
    if fund_ids is not None:
        s = s.where(t.c.SECODE.in_(fund_ids))

    df = pd.read_sql(s, db, index_col=['fund_id', 'stock_id'])

    return df


if __name__ == "__main__":
    load_fd_skdetail_all()
    load_fd_skdetail_ten()
