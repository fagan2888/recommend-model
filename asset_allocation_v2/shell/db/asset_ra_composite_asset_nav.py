#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
from . import database
import MySQLdb
import config

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# base.ra_index_nav
#
def load_nav(id_):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_composite_asset_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_asset_id,
        t1.c.ra_date,
        t1.c.ra_nav,
        t1.c.ra_inc,
    ]

    s = select(columns).where(t1.c.ra_asset_id == id_)

    df = pd.read_sql(s, db, index_col = ['ra_asset_id', 'ra_date'], parse_dates=['ra_date'])

    return df
