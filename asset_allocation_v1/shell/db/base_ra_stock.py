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
def find(globalid):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_stock', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.sk_code,
        t.c.sk_name,
    ]

    s = select(columns).where(t.c.globalid == globalid)

    return s.execute().first()



def stock_info(codes = None):

    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_stock', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.sk_code,
        t.c.sk_name,
    ]

    if codes is None:
        s = select(columns)
    else:
        s = select(columns).where(t.c.sk_code.in_(set(codes)))

    df = pd.read_sql(s, db, index_col = ['globalid'])

    return df
