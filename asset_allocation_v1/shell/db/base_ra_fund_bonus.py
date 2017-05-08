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

def load(gids=None, codes=None, sdate=None, edate=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_fund_bonus', metadata, autoload=True)

    columns = [
        t.c.ra_fund_id,
        t.c.ra_fund_code,
        t.c.ra_record_date,
        t.c.ra_dividend_date,
        t.c.ra_payment_date,
        t.c.ra_bonus,
        t.c.ra_bonus_nav,
        t.c.ra_bonus_nav_date,
    ]

    s = select(columns).order_by(t.c.ra_record_date.asc(), t.c.ra_fund_id.asc())
    if gids is not None:
        s = s.where(t.c.ra_fund_id.in_(gids))

    if codes is not None:
        s = s.where(t.c.ra_code.in_(codes))

    if sdate is not None:
        s = s.where(t.c.ra_record_date >= sdate)

    if edate is not None:
        s = s.where(t.c.ra_record_date <= edate)

    df = pd.read_sql(s, db, index_col=['ra_record_date', 'ra_fund_id'], parse_dates=['ra_record_date', 'ra_dividend_date', 'ra_payment_date', 'ra_bonus_nav_date'])

    return df

