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
# tc_timing
#
def load(id_, xtypes=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_bl', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.bl_name,
        t1.c.bl_type,
        t1.c.bl_method,
    ]

    s = select(columns)
    if id_ is not None:
        s = s.where(t1.c.globalid.in_(id_))
    if xtypes is not None:
        s = s.where(t1.c.tc_type.in_(xtypes))

    s = s.where(t1.c.bl_method != 0)
    df = pd.read_sql(s, db)

    return df

