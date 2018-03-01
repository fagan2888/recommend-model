#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database
import MySQLdb
import config

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# base.ra_index_nav
#
def load_series(id_, layer, reindex=None, begin_date=None, end_date=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('barra_stock_factor_layer_nav', metadata, autoload=True)

    columns = [
        t1.c.trade_date.label('date'),
        t1.c.nav.label('nav'),
    ]

    s = select(columns).where(t1.c.bf_id == id_).where(t1.c.layer == layer)
    
    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['nav']