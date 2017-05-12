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

def load(gids=None, codes=None, xtype=5):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('fund_fee', metadata, autoload=True)

    columns = [
        t.c.ff_fund_id,
        t.c.ff_code,
        # t.c.ff_type,
        # t.c.ff_min_value,
        t.c.ff_max_value,
        # t.c.ff_min_value_equal,
        # t.c.ff_max_value_equal,
        # t.c.ff_value_type,
        t.c.ff_fee,
        t.c.ff_fee_type,
    ]

    s = select(columns).where(t.c.ff_type == xtype)
    if gids is not None:
        s = s.where(t.c.ff_fund_id.in_(gids))

    if codes is not None:
        s = s.where(t.c.ff_code.in_(codes))

    df = pd.read_sql(s, db, index_col=['ff_fund_id'])

    df['ff_max_value'].fillna(np.inf, inplace=True)

    return df

def load_buy(gids=None, codes=None):
    return load(gids=gids, codes=codes, xtype=5)

def load_redeem(gids=None, codes=None):
    return load(gids=gids, codes=codes, xtype=6)

