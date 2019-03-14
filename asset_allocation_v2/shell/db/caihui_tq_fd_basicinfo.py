#coding=utf-8
'''
Created on: Dec. 28, 2018
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_fund_code_info(fund_ids=None, fund_codes=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_fd_basicinfo', metadata, autoload=True)

    columns = [
            t.c.SECODE.label('fund_id'),
            t.c.FSYMBOL.label('fund_code')
    ]

    s = select(columns)
    if fund_ids is not None:
        s = s.where(t.c.SECODE.in_(fund_ids))
    if fund_codes is not None:
        s = s.where(t.c.FSYMBOL.in_(fund_codes))

    df = pd.read_sql(s, engine, index_col=['fund_id'])

    return df


if __name__ == '__main__':

    load_fund_code_info()
    load_fund_code_info(fund_ids=['1030000001', '1030000002'])
    load_fund_code_info(fund_codes=['000001', '000003'])
