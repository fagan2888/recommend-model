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
        connections[key] = con
        return con

    return None

def asset_tc_timing_signal_load(timings, begin_date=None, end_date=None):
    db = connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('tc_timing_signal', metadata, autoload=True)

    columns = [
        t1.c.tc_date,
        t1.c.tc_timing_id,
        t1.c.tc_signal,
    ]

    s = select(columns).where(t1.c.tc_timing_id.in_(timings));
    
    if begin_date is not None:
        s = s.where(t1.c.tc_date >= begin_date)

    if end_date is not None:
        s = s.where(t1.c.tc_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['tc_date', 'tc_timing_id'], parse_dates=['tc_date'])

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df
    
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

def asset_rm_risk_mgr_pos_load(
        riskmgr_id, categories=None, begin_date = None, end_date=None):
    db = connection('asset')

    t1 = Table('rm_risk_mgr_pos', MetaData(bind=db), autoload=True)

    columns = [
        t1.c.rm_category,
        t1.c.rm_date,
        t1.c.rm_pos,
    ]

    s = select(columns).where(t1.c.rm_risk_mgr_id == riskmgr_id);

    if categories is not None:
        if hasattr(categories, "__iter__") or not isinstance(categories, str):
            s = s.where(t1.c.rm_category.in_(categories))
        else:
            s = s.where(t1.c.rm_category == categories)
    
    if begin_date is not None:
        s = s.where(t1.c.rm_date >= begin_date)

    if end_date is not None:
        s = s.where(t1.c.rm_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['rm_date', 'rm_category'], parse_dates=['rm_date'])

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df
    

    
