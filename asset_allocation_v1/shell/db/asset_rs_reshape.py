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

def find(globalid):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('rs_reshape', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.rs_type,
        t.c.rs_asset_id,
        t.c.rs_pool,
        t.c.rs_asset,
        t.c.rs_timing_id,
        t.c.rs_name,
    ]

    s = select(columns).where(t.c.globalid == globalid)

    return s.execute().first()
#
# mz_reshape
#
def load(gids, xtypes=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('rs_reshape', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.rs_type,
        t1.c.rs_asset_id,
        t1.c.rs_pool,
        t1.c.rs_asset,
        t1.c.rs_timing_id,
        t1.c.rs_start_date,
        t1.c.rs_name,
    ]

    s = select(columns)

    if gids is not None:
        s = s.where(t1.c.globalid.in_(gids))
    if xtypes is not None:
        s = s.where(t1.c.rs_type.in_(xtypes))
    
    df = pd.read_sql(s, db)

    return df

def max_id_between(min_id, max_id):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('rs_reshape', metadata, autoload=True)

    columns = [ t.c.globalid ]

    s = select([func.max(t.c.globalid).label('maxid')]).where(t.c.globalid.between(min_id, max_id))

    return s.execute().scalar()

