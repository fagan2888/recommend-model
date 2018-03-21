#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database
from ipdb import  set_trace

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# tc_timing
#
def load_argv(id_):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_bl_argv', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.bl_key,
        t1.c.bl_value,
    ]

    s = select(columns).where(t1.c.globalid == id_)

    df = pd.read_sql(s, db)
    argv = {}
    for _, kv in df.iterrows():
        argv[kv['bl_key']] = kv['bl_value']

    return argv

