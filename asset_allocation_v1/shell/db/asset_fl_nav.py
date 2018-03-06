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

def load_nav(id_):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('fl_nav', metadata, autoload=True)

    columns = [
        t1.c.fl_id,
        t1.c.fl_date,
        t1.c.fl_nav,
    ]

    s = select(columns).where(t1.c.fl_id == id_)

    df = pd.read_sql(s, db, index_col=['fl_id', 'fl_date'], parse_dates = ['fl_date'])

    return df