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
    t = Table('rm_riskmgr', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.rm_type,
        t.c.rm_algo,
        t.c.rm_asset_id,
        t.c.rm_timing_id,
        t.c.rm_start_date,
        t.c.rm_name,
    ]

    s = select(columns).where(t.c.globalid == globalid)

    return s.execute().first()
#
# mz_riskmgr
#
def load(gids, xtypes=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('rm_riskmgr', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.rm_type,
        t.c.rm_algo,
        t.c.rm_asset_id,
        t.c.rm_timing_id,
        t.c.rm_start_date,
        t.c.rm_name,
    ]

    s = select(columns)

    if gids is not None:
        s = s.where(t.c.globalid.in_(gids))
    if xtypes is not None:
        s = s.where(t.c.rm_type.in_(xtypes))
    
    df = pd.read_sql(s, db)

    return df

def where_asset_id(asset_id):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('rm_riskmgr', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.rm_type,
        t.c.rm_algo,
        t.c.rm_asset_id,
        t.c.rm_timing_id,
        t.c.rm_start_date,
        t.c.rm_name,
    ]

    s = select(columns)

    if asset_id is not None:
        s = s.where(t.c.rm_asset_id == asset_id)
    
    df = pd.read_sql(s, db)

    return df


# def max_id_between(min_id, max_id):
#     db = database.connection('asset')
#     metadata = MetaData(bind=db)
#     t = Table('rm_riskmgr', metadata, autoload=True)

#     columns = [ t.c.globalid ]

#     s = select([func.max(t.c.globalid).label('maxid')]).where(t.c.globalid.between(min_id, max_id))

#     return s.execute().scalar()

