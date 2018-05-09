#coding=utf8

from sqlalchemy import MetaData, Table, select, func, literal_column
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
# tc_timing
#
def save(id_, df):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('ra_bl_view', metadata, autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]

    s = select(columns).where(t.c.globalid == id_)
    df_old = pd.read_sql(s, db, index_col = ['bl_date', 'globalid', 'bl_index_id'], parse_dates = ['bl_date'])
    database.batch(db, t, df, df_old, timestamp = True)


def load(id_, index_id):
    db = database.connection('asset')
    metadata = MetaData(bind =  db)
    t = Table('ra_bl_view', metadata, autoload = True)
    columns = [
        t.c.bl_date,
        t.c.bl_view,
    ]

    s = select(columns).where(t.c.globalid == id_).where(t.c.bl_index_id == index_id)
    df = pd.read_sql(s, db, index_col = ['bl_date'], parse_dates = True)

    return df
