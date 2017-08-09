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

def find(globalid):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_fund', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.ra_code,
        t.c.ra_name,
        t.c.ra_type,
        t.c.ra_type_calc,
        t.c.ra_regtime,
        t.c.ra_volume,
    ]

    s = select(columns).where(t.c.globalid == globalid)

    return s.execute().first()

def load(globalids=None, codes=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_fund', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.ra_code,
        t.c.ra_name,
        t.c.ra_type,
        t.c.ra_type_calc,
        t.c.ra_regtime,
        t.c.ra_volume,
    ]

    s = select(columns)
    if globalids is not None:
        s = s.where(t.c.globalid.in_(globalids))

    if codes is not None:
        s = s.where(t.c.ra_code.in_(codes))

    df = pd.read_sql(s, db)

    return df


def find_type_fund(ra_type):

    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_fund', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.ra_code,
        t.c.ra_name,
        t.c.ra_type,
        t.c.ra_type_calc,
        t.c.ra_regtime,
        t.c.ra_volume,
    ]

    s = select(columns).where(t.c.ra_type == ra_type)

    df = pd.read_sql(s, db)

    return df
