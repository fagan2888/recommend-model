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

def load(uid):
    db = database.connection('trade')
    metadata = MetaData(bind=db)
    t = Table('ts_dividend_fund', metadata, autoload=True)

    columns = [
        t.c.ts_uid,
        t.c.ts_portfolio_id,
        t.c.ts_fund_code,
        t.c.ts_pay_method,
        t.c.ts_record_date,
        t.c.ts_dividend_date,
        t.c.ts_dividend_amount,
        t.c.ts_dividend_share,
    ]

    s = select(columns).where(t.c.ts_uid == uid)

    df = pd.read_sql(s, db, parse_dates=['ts_record_date', 'ts_dividend_date'])

    return df

