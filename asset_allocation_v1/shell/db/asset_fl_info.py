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

def load():
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('fl_info', metadata, autoload=True)

    columns = [
        t1.c.fl_id,
        t1.c.fl_asset_id,
        t1.c.fl_first_loc,
        t1.c.fl_second_loc,
    ]

    s = select(columns)

    df = pd.read_sql(s, db, index_col=['fl_id', 'fl_asset_id'])

    return df