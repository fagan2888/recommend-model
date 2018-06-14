#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import numpy as np
import pandas as pd
# import os
# import sys
import logging
from . import database
# import pdb

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

    df = pd.read_sql(s, db, parse_dates=['ra_record_date', 'ra_dividend_date', 'ra_payment_date', 'ra_bonus_nav_date'])

    #
    # QDII基金的除息日经常在权益登记日之前，但实际上，QDII的权益登记日
    # 登记的是除息日的份额，为了计算方便， 如果权益登记日 > 除息日，则
    # 将除息日设置为权益登记日。
    #
    # pdb.set_trace()
    df.loc[df['ra_record_date'] > df['ra_dividend_date'], 'ra_record_date'] = df['ra_dividend_date']
    df = df.set_index(['ra_record_date', 'ra_fund_id']).sort_index()

    return df

