#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
from . import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# on_online
#
def load(gids):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('on_online_alloc', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.on_risk,
        t1.c.on_online_id,
        t1.c.on_name,
    ]

    s = select(columns)

    if gids is not None:
        s = s.where(t1.c.globalid.in_(gids))
    
    df = pd.read_sql(s, db)

    return df

def where_online_id(online_id):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('on_online_alloc', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.on_risk,
        t1.c.on_online_id,
        t1.c.on_name,
    ]

    s = select(columns)

    if online_id is not None:
        s = s.where(t1.c.on_online_id == online_id)
    
    df = pd.read_sql(s, db)

    return df

