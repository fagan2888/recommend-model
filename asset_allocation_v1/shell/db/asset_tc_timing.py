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
def load(timings, xtypes=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('tc_timing', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.tc_type,
        t1.c.tc_method,
        t1.c.tc_index_id,
        t1.c.tc_begin_date,
        t1.c.tc_argv,
        t1.c.tc_name,
    ]

    s = select(columns)
    if timings is not None:
        s = s.where(t1.c.globalid.in_(timings))
    if xtypes is not None:
        s = s.where(t1.c.tc_type.in_(xtypes))

    s = s.where(t1.c.tc_method != 0)
    df = pd.read_sql(s, db)

    return df

def max_id_between(min_id, max_id):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('tc_timing', metadata, autoload=True)

    columns = [ t.c.globalid ]

    s = select([func.max(t.c.globalid).label('maxid')]).where(t.c.globalid.between(min_id, max_id))

    return s.execute().scalar()
