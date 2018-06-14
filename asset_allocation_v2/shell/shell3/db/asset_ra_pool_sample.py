#coding=utf8

from sqlalchemy import MetaData, Table, select, func, literal_column
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
from . import database
from . import asset_ra_pool

from dateutil.parser import parse

logger = logging.getLogger(__name__)

def load(gid):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_pool_sample', metadata, autoload=True)

    columns = [
        t1.c.ra_fund_code,
    ]

    s = select(columns)

    if gid is not None:
        s = s.where(t1.c.ra_pool_id == gid)
    else:
        return None

    df = pd.read_sql(s, db)

    return df
