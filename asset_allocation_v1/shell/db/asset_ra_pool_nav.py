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
# asset.ra_pool_nav
#
def load_series(id_, category, xtype, reindex=None, begin_date=None, end_date=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_pool_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date.label('date'),
        t1.c.ra_nav.label('nav'),
    ]

    s = select(columns) \
        .where(t1.c.ra_pool == id_) \
        .where(t1.c.ra_category == category) \
        .where(t1.c.ra_type == xtype)
    
    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex)

    return df['nav']

