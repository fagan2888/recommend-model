#coding=utf-8
'''
Edited at Dec. 28, 2018
Editor: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_index_basic_info(index_ids=None, index_codes=None, est_class=None, status=1):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_ix_basicinfo', metadata, autoload=True)

    columns = [
            t.c.SECODE.label('index_id'),
            t.c.SYMBOL.label('index_code')
    ]

    s = select(columns)
    if index_ids is not None:
        s = s.where(t.c.SECODE.in_(index_ids))
    if index_codes is not None:
        s = s.where(t.c.SYMBOL.in_(index_codes))
    if est_class is not None:
        s = s.where(t.c.ESTCLASS==est_class)
    if status is not None:
        s = s.where(t.c.STATUS==status)

    df = pd.read_sql(s, engine, index_col=['index_id'])

    return df


if __name__ == '__main__':

    load_index_basic_info(est_class='申万一级行业指数')

