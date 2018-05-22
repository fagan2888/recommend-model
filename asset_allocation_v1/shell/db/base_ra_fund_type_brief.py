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

def load_fund_type(globalid):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_fund_type_brief', metadata, autoload=True)

    columns = [
        t.c.ra_fund_id,
        t.c.ra_fund_type,
    ]

    sql = select(columns).where(t.c.ra_fund_id.in_(globalid))
    df = pd.read_sql(sql, db, index_col = ['ra_fund_id'])

    return df

