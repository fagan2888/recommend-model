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

def format(df, columns=[], fmter=None, kwcolumns=[]):
    if columns and fmter:
        for column in columns:
            if column in df.columns and column not in kwcolumns:
                df[column] = df[column].map(fmter)

    if kwcolumns:
        for k, v in kwcolumns:
            if v and k in df.columns:
                df[k] = df[k].map(v)
    return df

def number_format(df, columns=[], precision=2, kwcolumns=[]):
    if columns:
        for column in columns:
            if column in df.columns and column not in kwcolumns:
                df[column] = df[column].map(("{:.%df}" % (precision)).format)

    if kwcolumns:
        for k, v in kwcolumns:
            if v and k in df.columns:
                df[k] = df[k].map(("{:.%df}" % (v)).format)
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

        df_insert.to_sql(table.name, db, index=True, if_exists='append', chunksize=500)

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
#
# tc_timing
#
def asset_tc_timing_load(timings):
    db = connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('tc_timing', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.tc_type,
        t1.c.tc_method,
        t1.c.tc_index_id,
        t1.c.tc_begin_date,
        t1.c.tc_name,
    ]

    s = select(columns)

    if timings is not None:
        s = s.where(t1.c.globalid.in_(timings))
    
    df = pd.read_sql(s, db)

    return df

def asset_tc_timing_scratch_load_signal(timings):
    db = connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('tc_timing_scratch', metadata, autoload=True)

    columns = [
        t1.c.tc_timing_id,
        t1.c.tc_date,
        t1.c.tc_signal,
    ]

    s = select(columns)

    if timings is not None:
        if hasattr(timings, "__iter__") and not isinstance(timings, str):
            s = s.where(t1.c.tc_timing_id.in_(timings))
        else:
            s = s.where(t1.c.tc_timing_id == timings)
    
    df = pd.read_sql(s, db, index_col = ['tc_date', 'tc_timing_id'], parse_dates=['tc_date'])

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df


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
        if hasattr(categories, "__iter__") and not isinstance(categories, str):
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
    

#
# base.trade_dates
#
def base_trade_dates_load_index(begin_date=None, end_date=None):
    db = connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('trade_dates', metadata, autoload=True)

    columns = [
        t1.c.td_date,
        t1.c.td_type,
    ]

    s = select(columns)
    if begin_date is not None:
        s = s.where(t1.c.td_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.td_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['td_date'], parse_dates=['td_date'])

    return df.index

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
        if hasattr(categories, "__iter__") and not isinstance(categories, str):
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
    

    
    if begin_date is not None:
        s = s.where(t1.c.td_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.td_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['td_date'], parse_dates=['td_date'])

    return df.index

#
# base.ra_fund
#
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

#
# base.ra_fund_nav
#
def base_ra_fund_nav_load_weekly(begin_date, end_date, fund_ids=None, codes=None):
    db = connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_fund_nav', metadata, autoload=True)
    t2 = Table('trade_dates', metadata, autoload=True)

    columns = [
        t1.c.ra_code.label('code'),
        t1.c.ra_date.label('date'),
        t1.c.ra_nav_adjusted,
    ]

    s = select(columns) \
        .select_from(t1.join(t2, t1.c.ra_date == t2.c.td_date)) \
        .where(t1.c.ra_date.between(begin_date, end_date)) \
        .where(t2.c.td_date.between(begin_date, end_date) & (t2.c.td_type.op('&')(0x02) | (t2.c.td_date == end_date)))
    
    if fund_ids is not None:
        s = s.where(t1.c.ra_fund_id.in_(fund_ids))

    if codes is not None:
        s = s.where(t1.c.ra_code.in_(codes))

    df = pd.read_sql(s, db, index_col = ['date', 'code'], parse_dates=['date'])

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df

def base_ra_fund_nav_load_daily(begin_date, end_date, fund_ids=None, codes=None):
    db = connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_fund_nav', metadata, autoload=True)
    t2 = Table('trade_dates', metadata, autoload=True)

    columns = [
        t1.c.ra_code.label('code'),
        t1.c.ra_date.label('date'),
        t1.c.ra_nav_adjusted,
    ]

    s = select(columns) \
        .select_from(t1.join(t2, t1.c.ra_date == t2.c.td_date)) \
        .where(t1.c.ra_date.between(begin_date, end_date)) \
        .where(t2.c.td_date.between(begin_date, end_date))
    
    if fund_ids is not None:
        s = s.where(t1.c.ra_fund_id.in_(fund_ids))

    if codes is not None:
        s = s.where(t1.c.ra_code.in_(codes))

    df = pd.read_sql(s, db, index_col = ['date', 'code'], parse_dates=['date'])

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df

def base_ra_fund_nav_load_series(code, reindex=None, begin_date=None, end_date=None):
    db = connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_fund_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date.label('date'),
        t1.c.ra_nav_adjusted.label('nav'),
    ]

    s = select(columns) \
        .where((t1.c.ra_code == code) | (t1.c.ra_fund_id == code))

    # s = select(columns).where(t1.c.ra_fund_id == id_)
    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex)

    return df['nav']

#
# base.ra_index
#
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
#
# base.ra_index_nav
#
def base_ra_index_nav_load_series(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_index_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date.label('date'),
        t1.c.ra_nav.label('nav'),
    ]

    s = select(columns).where(t1.c.ra_index_id == id_)
    
    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
    if mask is not None:
        s = s.where(t1.c.ra_mask == mask)
        
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex)

    return df['nav']
#
# asset.allocation_instance
#
def asset_allocation_instance_new_globalid(xtype=1, replace=False):
    db = connection('asset')
    metadata = MetaData(bind=db)
    t = Table('allocation_instances', metadata, autoload=True)

    today = datetime.now().strftime("%Y%m%d")
    if xtype == 9:
        between_min, between_max = ('%s90' % (today), '%s99' % (today))
    else:
        between_min, between_max = ('%s00' % (today), '%s89' % (today))

    columns = [ t.c.id ]

    s = select([func.max(t.c.id).label('maxid')]).where(t.c.id.between(between_min, between_max))

    max_id = s.execute().scalar()

    if max_id is None:
        ret = int(between_min)
    else:
        if max_id >= between_max:
            logger.warning("run out of allocation instance id [%s]!", max_id)
            ret = None
        else:
            if replace:
                ret = max_id
            else:
                ret = max_id + 1
    return ret

#
# asset.allocation_instance_nav
#
def asset_allocation_instance_nav_load(inst, xtype, allocs=None, begin=None, end=None):
    db = connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('allocation_instance_nav', metadata, autoload=True)
    t2 = Table('trade_dates', metadata, autoload=True)

    columns = [
        t1.c.ai_inst_id,
        t1.c.ai_alloc_id,
        t1.c.ai_date,
        t1.c.ai_nav,
    ]

    s = select(columns) \
        .select_from(t1.join(t2, t1.c.ai_date == t2.c.td_date)) \
        .where(t1.c.ai_inst_id == inst) \
        .where(t1.c.ai_type == xtype);
    if allocs is not None:
        s = s.where(t1.c.ai_alloc_id.in_(allocs))
    if begin is not None:
        s = s.where(t1.c.ai_date >= begin)
    if end is not None:
        s = s.where(t1.c.ai_date <= end)
        
    df = pd.read_sql(s, db, index_col = ['ai_inst_id', 'ai_date', 'ai_alloc_id'], parse_dates=['ai_date'])

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df

def asset_allocation_instance_nav_load_series(
        id_, alloc_id, xtype, reindex=None, begin_date=None, end_date=None):
    db = connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('allocation_instance_nav', metadata, autoload=True)

    columns = [
        t1.c.ai_date.label('date'),
        t1.c.ai_nav.label('nav'),
    ]

    s = select(columns) \
        .where(t1.c.ai_inst_id == id_) \
        .where(t1.c.ai_alloc_id == alloc_id) \
        .where(t1.c.ai_type == xtype)
    
    if begin_date is not None:
        s = s.where(t1.c.ai_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ai_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex)

    return df['nav']

def asset_allocation_instance_position_detail_load(id_):
    db = connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('allocation_instance_position_detail', metadata, autoload=True)

    columns = [
        t1.c.ai_inst_type,
        t1.c.ai_alloc_id,
        t1.c.ai_transfer_date,
        t1.c.ai_category,
        t1.c.ai_fund_id,
        t1.c.ai_fund_code,
        t1.c.ai_fund_ratio,
    ]

    s = select(columns).where(t1.c.ai_inst_id == id_)
        
    df = pd.read_sql(s, db, index_col = ['ai_alloc_id', 'ai_transfer_date', 'ai_category', 'ai_fund_id'], parse_dates=['ai_transfer_date'])

    return df
    

#
# asset.ra_composite_asset_nav
#
def asset_ra_composite_asset_load_series(id_, reindex=None, begin_date=None, end_date=None):
    db = connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_composite_asset_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date.label('date'),
        t1.c.ra_nav.label('nav'),
    ]

    s = select(columns).where(t1.c.ra_asset_id == id_)
    
    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex)

    return df['nav']

#
# asset.ra_composite_asset_position
#
def asset_ra_composite_asset_position_load(asset_id):
    db = connection('asset')
    
    # 加载基金列表
    t = Table('ra_composite_asset_position', MetaData(bind=db), autoload=True)
    columns = [
        t.c.ra_date,
        t.c.ra_fund_code,
        t.c.ra_fund_ratio,
    ]
    s = select(columns, (t.c.ra_asset_id == asset_id))
    
    df = pd.read_sql(s, db, index_col = ['ra_date', 'ra_fund_code'], parse_dates=['ra_date'])

    return df

#
# asset.ra_pool_nav
#
def asset_ra_pool_nav_load_series(id_, category, xtype, reindex=None, begin_date=None, end_date=None):
    db = connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_pool_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date.label('date'),
        t1.c.ra_nav.label('nav'),
    ]

    s = select(columns) \
        .where(t1.c.ra_pool == id_) \
        .where(t1.c.ra_category == category) \
        .where(t1.c.ra_type == xtype)
    
    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex)

    return df['nav']

#
# asset.risk_asset_allocation_nav
#
def asset_risk_asset_allocation_nav_load_series(
        alloc_id, xtype, reindex=None, begin_date=None, end_date=None):
    db = connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('risk_asset_allocation_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date.label('date'),
        t1.c.ra_nav.label('nav'),
    ]

    s = select(columns) \
        .where(t1.c.ra_alloc_id == alloc_id) \
        .where(t1.c.ra_type == xtype)
    
    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
        
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex)

    return df['nav']
