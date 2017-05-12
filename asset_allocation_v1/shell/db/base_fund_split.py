#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import numpy as np
import pandas as pd
# import os
# import sys
import logging
import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

def load(gids=None, sdate=None, edate=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('fund_split', metadata, autoload=True)

    columns = [
        t.c.fs_fund_id.label('ra_fund_id'),
        # t.c.ra_fund_code,
        t.c.fs_split_date.label('ra_split_date'),
        t.c.fs_split_proportion.label('ra_split_proportion'),
    ]

    s = select(columns)
    if gids is not None:
        s = s.where(t.c.fs_fund_id.in_(gids))

    if sdate is not None:
        s = s.where(t.c.fs_split_date >= sdate)

    if edate is not None:
        s = s.where(t.c.fs_split_date <= edate)

    df = pd.read_sql(s, db, index_col=['ra_split_date', 'ra_fund_id'], parse_dates=['ra_split_date'])

    return df

