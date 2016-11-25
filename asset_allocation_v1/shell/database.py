#coding=utf8


import string
import MySQLdb
import config
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import sys
import logging
sys.path.append('shell')
import Const

from sqlalchemy import *


from dateutil.parser import parse

logger = logging.getLogger(__name__)

uris = {
    'asset': config.db_asset_uri,
    'base': config.db_base_uri,
}

connections = {}

def connection(key):
    if key in connections:
        return connections[key]

    if key in uris:
        uri = uris[key]
        con = create_engine(uri)
        # con.echo = True
        connections[key] = con
        return con

    return None

def batch(db, table, df_new, df_old, timestamp=True):
    index_insert = df_new.index.difference(df_old.index)
    index_delete = df_old.index.difference(df_new.index)
    index_update = df_new.index.intersection(df_old.index)

    if len(index_delete):
        keys = [table.c.get(c) for c in df_new.index.names]
        table.delete(tuple_(*keys).in_(index_delete.tolist())).execute()

        if len(index_delete) > 1:
            logger.info("delete %s (%5d) : %s " % (table.name, len(index_insert), index_delete[0]))
        
        
    if len(index_insert):
        df_insert = df_new.loc[index_insert].copy()
        if timestamp:
            df_insert['updated_at'] = df_insert['created_at'] = datetime.now()

        df_insert.to_sql(table.name, db, index=True, if_exists='append')

        if len(index_insert) > 1:
            logger.info("insert %s (%5d) : %s " % (table.name, len(index_insert), index_insert[0]))

    if len(index_update):
        df1 = df_new.loc[index_update].copy()
        df2 = df_old.loc[index_update].copy()

        masks = (df1 != df2)
        df_update = df1.loc[masks.any(axis=1)].copy()

        for key, row in df_update.iterrows():
            origin = df2.loc[key]
            columns = row[masks.loc[key]]

            pkeys = zip(df_update.index.names, key)

            dirty = {k:{'old':origin[k], 'new':v} for k,v in columns.iteritems()}

            if timestamp:
                columns['updated_at'] = text('NOW()')

            values = {k:v for k,v in columns.iteritems()}
            stmt = table.update(values=values)
            for k, v in pkeys:
                stmt = stmt.where(table.c.get(k) == v)

            logger.info("update %s : {key: %s, dirties: %s " % (table.name, str(pkeys), str(dirty)))
            stmt.execute()

def base_ra_fund_find(globalid):
    db = connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_fund', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.ra_code,
        t.c.ra_name,
        t.c.ra_type,
        t.c.ra_type_calc,
        t.c.ra_regtime,
        t.c.ra_volume,
    ]

    s = select(columns).where(t.c.globalid == globalid)

    return s.execute().first()

def base_ra_fund_load(globalids=None, codes=None):
    db = connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_fund', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.ra_code,
        t.c.ra_name,
        t.c.ra_type,
        t.c.ra_type_calc,
        t.c.ra_regtime,
        t.c.ra_volume,
    ]

    s = select(columns)
    if globalids is not None:
        s = s.where(t.c.globalid.in_(globalids))

    if codes is not None:
        s = s.where(t.c.ra_code.in_(codes))

    df = pd.read_sql(s, db)

    return df

def base_ra_index_find(globalid):
    db = connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_index', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.ra_code,
        t.c.ra_name,
        t.c.ra_announce_date,
        t.c.ra_begin_date,
        t.c.ra_base_date,
    ]

    s = select(columns).where(t.c.globalid == globalid)

    return s.execute().first()

