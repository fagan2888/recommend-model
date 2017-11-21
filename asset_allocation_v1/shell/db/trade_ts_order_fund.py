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

def load(uid, xtype=None):
    db = database.connection('trade')
    metadata = MetaData(bind=db)
    t = Table('ts_order_fund', metadata, autoload=True)

    columns = [
        t.c.ts_txn_id,
        t.c.ts_uid,
        t.c.ts_portfolio_id,
        t.c.ts_pay_method,
        t.c.ts_trade_type,
        t.c.ts_trade_status,
        t.c.ts_placed_date,
        t.c.ts_placed_time,
        t.c.ts_placed_amount,
        t.c.ts_placed_share,
        t.c.ts_acked_date,
        t.c.ts_acked_amount,
        t.c.ts_acked_share,
        t.c.ts_acked_fee,
        t.c.ts_portfolio_txn_id,
    ]

    s = select(columns).where(t.c.ts_uid == uid)
    if xtype is not None:
        s = s.where(t.c.ts_trade_type.in_(xtype))

    # if codes is not None:
    #     s = s.where(t.c.ff_code.in_(codes))

    df = pd.read_sql(s, db)

    return df

