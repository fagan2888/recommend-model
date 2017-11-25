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
# base.ra_index
#
def find(globalid):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_stock', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.ra_code,
        t.c.ra_name,
    ]

    s = select(columns).where(t.c.globalid == globalid)

    return s.execute().first()
