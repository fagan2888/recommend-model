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

def load_ack(gids=None, codes=None, xtype=5):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('fund_infos', metadata, autoload=True)

    columns = [
        t.c.fi_globalid,
        t.c.fi_code,
        t.c.fi_yingmi_confirm_time,
        t.c.fi_yingmi_to_account_time
    ]

    s = select(columns)
    if gids is not None:
        s = s.where(t.c.fi_globalid.in_(gids))

    if codes is not None:
        s = s.where(t.c.fi_code.in_(codes))

    df = pd.read_sql(s, db, index_col=['fi_globalid'])

    df['buy'] = df['fi_yingmi_confirm_time'] + 1 # 实际中，购买完成，份额可赎回时间是订单确认时间 + 1天
    sr = df['fi_yingmi_to_account_time'] - 2
    df['redeem'] = sr.where(sr >= 1, 1)
    df['code'] = df['fi_code'].apply(lambda x: "%06d" % x)

    return df[['code', 'buy', 'redeem']]

