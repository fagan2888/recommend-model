#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# base.ra_index
#
def find(ts_date):
    db = database.connection('trade')
    metadata = MetaData(bind=db)
    t = Table('ts_holding_fund', metadata, autoload=True)

    columns = [
        t.c.ts_fund_code,
        t.c.ts_date,
        t.c.ts_amount,
    ]

    s = select(columns).where(t.c.ts_date == ts_date)
    df = pd.read_sql(s, db)

    return df


def load():
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_index', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.ra_name,
    ]

    s = select(columns)
    df = pd.read_sql(s, db, index_col = ['globalid'])

    return df
