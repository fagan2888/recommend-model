#coding=utf8


from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
import re
# import os
# import sys
import logging
import database
from db import base_ra_index
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


from dateutil.parser import parse

logger = logging.getLogger(__name__)

Base = declarative_base()


class ra_fund(Base):

    __tablename__ = 'ra_fund'

    globalid = Column(Integer, primary_key = True)

    ra_code = Column(String)
    ra_name = Column(String)

    ra_type = Column(String)

    ra_fund_type = Column(Integer)


    updated_at = Column(DateTime)
    created_at = Column(DateTime)



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


def load_all_globalid():
    db = database.connection('base')
    metadata = MetaData(bind = db)
    t = Table('ra_fund', metadata, autoload=True)

    columns = [
        t.c.globalid,
    ]

    s = select(columns)
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

    s = select(columns).where(t.c.ra_type == ra_type).where(t.c.ra_mask == 0)

    df = pd.read_sql(s, db)

    return df

if __name__ == '__main__':
    print load_all_globalid()
