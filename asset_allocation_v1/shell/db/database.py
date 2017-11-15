#coding=utf8


import sys
sys.path.append('./shell')
import string
import MySQLdb
import config
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import logging
import re
import Const

from sqlalchemy import *
from sqlalchemy import event
from sqlalchemy import exc
from sqlalchemy.engine import Engine as engine
from sqlalchemy.pool import Pool
from util.xlist import chunks
from util.xdebug import dd

from dateutil.parser import parse

import asset_mz_highlow_alloc
import asset_mz_highlow_asset
import asset_mz_highlow_pos
import asset_mz_markowitz_asset
import asset_mz_markowitz_pos
import asset_ra_pool
import asset_ra_pool_nav
import asset_rs_reshape
import asset_rs_reshape_nav
import base_ra_fund
import base_ra_fund_nav
import base_ra_index
import base_ra_index_nav
import base_exchange_rate_index_nav
import base_exchange_rate_index
import MySQLdb
from DBUtils.PooledDB import PooledDB

logger = logging.getLogger(__name__)

uris = {
    'asset': config.db_asset_uri,
    'base': config.db_base_uri,
    'caihui': config.db_caihui_uri,
    'trade': config.db_trade_uri,
    'portfolio_sta': config.db_portfolio_sta_uri,
}

connections = {}

def connection(key):
    if key in connections:
        return connections[key]

    if key in uris:
        uri = uris[key]
        con = create_engine(uri)
        #con.echo = True
        connections[key] = con
        return con

    return None


@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    connection_record.info['pid'] = os.getpid()


@event.listens_for(engine, "checkout")
def checkout(dbapi_connection, connection_record, connection_proxy):
    pid = os.getpid()
    if connection_record.info['pid'] != pid:
        connection_record.connection = connection_proxy.connection = None
        raise exc.DisconnectionError(
            "Connection record belongs to pid %s, "
            "attempting to check out in pid %s" %
            (connection_record.info['pid'], pid)
        )

@event.listens_for(Pool, "checkout")
def ping_connection(dbapi_connection, connection_record, connection_proxy):
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SELECT 1")
    except:
        # optional - dispose the whole pool
        # instead of invalidating one at a time
        # connection_proxy._pool.dispose()
        # raise DisconnectionError - pool will try
        # connecting again up to three times before raising.
        raise exc.DisconnectionError()
        cursor.close()


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

def number_format(df, columns=[], precision=2, **kwcolumns):
    if columns:
        for column in columns:
            if column in df.columns and column not in kwcolumns:
                df[column] = df[column].map(("{:.%df}" % (precision)).format)

    if kwcolumns:
        for k, v in kwcolumns.iteritems():
            if v and k in df.columns:
                df[k] = df[k].map(("{:.%df}" % (v)).format)
    return df

def batch(db, table, df_new, df_old, timestamp=True):
    index_insert = df_new.index.difference(df_old.index)
    index_delete = df_old.index.difference(df_new.index)
    index_update = df_new.index.intersection(df_old.index)

    #print df_new.index
    #print df_old.index
    #print index_delete
    #
    # 我们首先计算需要更新的条目个数, 因为更新的性能很差, 所以, 如果需
    # 要更新的条目过多(>50), 则采用删除+插入的批量处理方式
    #
    if len(index_update):
        df1 = df_new.loc[index_update].copy()
        df2 = df_old.loc[index_update].copy()

        masks = (df1 != df2)
        df_update = df1.loc[masks.any(axis=1)].copy()
    else:
        df_update = pd.DataFrame(columns=df_new.columns)




    if len(df_update) > 50:
        keys = [table.c.get(c) for c in df_old.index.names]
        for segment in chunks(df_old.index.tolist(), 500):
            table.delete(tuple_(*keys).in_(segment)).execute()

        if timestamp:
            df_new['updated_at'] = df_new['created_at'] = datetime.now()

        df_new.to_sql(table.name, db, index=True, if_exists='append', chunksize=500)

        if len(df_new.index) >= 1:
            logger.info("delete %s (%5d records) and reinsert (%5d records): %s " % (table.name, len(df_old.index), len(df_new.index), df_new.index[0]))
    else:
        if len(index_delete):
            keys = [table.c.get(c) for c in df_new.index.names]
            for segment in chunks(index_delete.tolist(), 500):
                if len(df_update.index.names) == 1:
                    s = table.delete(keys[0].in_(segment))
                else:
                    s = table.delete(tuple_(*keys).in_(segment))
                #print s.compile(compile_kwargs={"literal_binds": True})
                s.execute()
            #table.delete(tuple_(*keys).in_(index_delete.tolist())).execute()

            if len(index_delete) > 1:
                logger.info("delete %s (%5d records) : %s " % (table.name, len(index_insert), index_delete[0]))


        if len(index_insert):
            df_insert = df_new.loc[index_insert].copy()
            if timestamp:
                df_insert['updated_at'] = df_insert['created_at'] = datetime.now()

            df_insert.to_sql(table.name, db, index=True, if_exists='append', chunksize=500)

            if len(index_insert) > 1:
                logger.info("insert %s (%5d records) : %s " % (table.name, len(index_insert), index_insert[0]))

        if len(df_update):
            for key, row in df_update.iterrows():
                origin = df2.loc[key]
                columns = row[masks.loc[key]]

                pkeys = []
                if len(df_update.index.names) == 1:
                    pkeys.append((df_update.index.names[0], key))
                else:
                    pkeys = zip(df_update.index.names, key)

                dirty = {k:{'old':origin[k], 'new':v} for k,v in columns.iteritems()}

                if timestamp:
                    columns['updated_at'] = text('NOW()')

                values = {k:v for k,v in columns.iteritems()}
                stmt = table.update(values=values)
                for k, v in pkeys:
                    stmt = stmt.where(table.c.get(k) == v)

                logger.info("update %s : {key: %s, dirties: %s} " % (table.name, str(pkeys), str(dirty)))
                stmt.execute()

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


def asset_rm_risk_mgr_signal_load(
        riskmgr_id, categories=None, begin_date = None, end_date=None):
    db = connection('asset')

    t1 = Table('rm_risk_mgr_signal', MetaData(bind=db), autoload=True)

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


def asset_rm_risk_mgr_signal_load(
        riskmgr_id, categories=None, begin_date = None, end_date=None):
    db = connection('asset')

    t1 = Table('rm_risk_mgr_signal', MetaData(bind=db), autoload=True)

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
        df = df.reindex(reindex, method='pad')

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
        df = df.reindex(reindex, method='pad')

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
        df = df.reindex(reindex, method='pad')

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
        df = df.reindex(reindex, method='pad')

    return df['nav']


def load_nav_series(asset_id, reindex=None, begin_date=None, end_date=None):

    if asset_id.isdigit():
        xtype = int(asset_id) / 10000000
    else:
        xtype = re.sub(r'([\d]+)','',asset_id).strip()

    if xtype == 1:
        #
        # 基金池资产
        #
        if asset_id.isdigit():
            asset_id = int(asset_id) % 10000000
            (pool_id, category) = (asset_id / 100, asset_id % 100)
            ttype = pool_id / 10000
        else:
            pool_id, category, ttype = asset_id, 0, 9
        sr = asset_ra_pool_nav.load_series(
            pool_id, category, ttype, reindex=reindex, begin_date=begin_date, end_date=end_date)
    elif xtype == 3:
        #
        # 基金池资产
        #
        sr = base_ra_fund_nav.load_series(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
    elif xtype == 4:
        #
        # 修型资产
        #
        sr = asset_rs_reshape_nav.load_series(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
    elif xtype == 12:
        #
        # 指数资产
        #
        sr = base_ra_index_nav.load_series(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
    elif xtype == 'ERI':

        sr = base_exchange_rate_index_nav.load_series(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
    else:
        sr = pd.Series()

    return sr


def load_asset_name_and_type(asset_id):
    (name, category) = ('', 0)

    if asset_id.isdigit():
        asset_id = int(asset_id)
        xtype = asset_id / 10000000
    else:
        xtype = re.sub(r'([\d]+)','',asset_id).strip()

    if xtype == 1:
        #
        # 基金池资产
        #
        asset_id %= 10000000
        (pool_id, category) = (asset_id / 100, asset_id % 100)
        ttype = pool_id / 10000
        name = asset_ra_pool.load_asset_name(pool_id, category, ttype)
    elif xtype == 3:
        #
        # 基金池资产
        #
        category = 1
        fund = base_ra_fund.find(asset_id)
        name = "%s(%s)" % (fund['ra_name'], fund['ra_code'])

    elif xtype == 4:
        #
        # 修型资产
        #
        asset = asset_rs_reshape.find(asset_id)
        (name, category) = (asset['rs_name'], asset['rs_asset'])
    elif xtype == 12:
        #
        # 指数资产
        #
        asset = base_ra_index.find(asset_id)
        name = asset['ra_name']
        if '标普' in name:
            category = 41
        elif '黄金' in name:
            category = 42
        elif '恒生' in name:
            category = 43
    elif xtype == 'ERI':
        asset = base_exchange_rate_index.find(asset_id)
        (name, category) = (asset['eri_name'], 0)

    else:
         (name, category) = ('', 0)

    return (name, category)

'''
def load_pool_via_asset(asset_id):
    xtype = asset_id / 10000000

    if asset_id.isdigit():
        asset_id = int(asset_id)
        xtype = asset_id / 10000000
    else:
        xtype = re.sub(r'([\d]+)','',asset_id).strip()


    if xtype == 12:
        #
        # 指数资产
        #
        asset = base_ra_index.find(asset_id)
        name = asset['ra_name']
        if '标普' in name:
            pool = 19240141
        elif '黄金' in name:
            pool = 19240142
        elif '恒生' in name:
            pool = 19240143
    else:
        pool = 0

    return pool
'''
def load_asset_and_pool(gid):
    gid = int(gid)
    xtype = gid / 10000000

    if xtype == 5:
        #
        # 马克维茨
        #
        df_asset = asset_mz_markowitz_asset.load([gid])
        df_asset = df_asset[['mz_asset_id', 'mz_asset_name', 'mz_asset_type']]
        df_asset.rename(columns={
            'mz_asset_id':'asset_id',
            'mz_asset_name':'asset_name',
            'mz_asset_type':'asset_type',
        }, inplace=True)
        df_asset['pool_id'] = df_asset['mz_asset_type']
        df_asset = df_asset.set_index(['asset_id'])

    elif xtype == 7:
        #
        # 高低风险
        #
        df_asset = asset_mz_highlow_asset.load([gid])
        df_asset = df_asset[['mz_asset_id', 'mz_asset_name', 'mz_asset_type', 'mz_pool_id']]
        df_asset.rename(columns={
            'mz_asset_id':'asset_id',
            'mz_asset_name':'asset_name',
            'mz_asset_type':'asset_type',
            'mz_pool_id':'pool_id',
        }, inplace=True)
        # df_asset = df_asset.set_index(['asset_id'])

    # elif xtype == 12:
    #     #
    #     # 指数资产
    #     #
    #     asset = base_ra_index.find(asset_id)
    #     name = asset['ra_name']
    #     if '标普' in name:
    #         category = 41
    #     elif '黄金' in name:
    #         category = 42
    #     elif '恒生' in name:
    #         category = 43
    else:
        df_asset = pd.DataFrame(columns=['asset_id', 'asset_name', 'asset_type'])

    return df_asset;

def load_alloc_and_risk(gid):
    gid = int(gid)
    xtype = gid / 10000000

    result = []
    if xtype == 5:
        #
        # 马克维茨
        #
        result.append((1.0, gid))

    elif xtype == 7:
        #
        # 基金池资产
        #
        df_asset = asset_mz_highlow_alloc.where_highlow_id(gid)
        for _,v in df_asset.iterrows():
            result.append((v['mz_risk'], v['globalid']))
    else:
        pass;

    return result

def load_pos_frame(gid):
    prefix = gid[0:2]
    if prefix.isdigit():
        gid = int(gid)
        xtype = gid / 10000000

        if xtype == 5:
            #
            # 马克维茨
            #
            df = asset_mz_markowitz_pos.load(gid)

        elif xtype == 7:
            #
            # 高低风险
            #
            df = asset_mz_highlow_pos.load(gid)
        else:
            df = pd.DataFrame(columns=['mz_date', 'mz_asset_id', 'mz_ratio'])

    else:
        if prefix == 'MZ':
            df = asset_mz_markowitz_pos.load(gid)
        elif prefix == 'HL':
            df = asset_mz_highlow_pos.load(gid)
        else:
            df = pd.DataFrame(columns=['mz_date', 'mz_asset_id', 'mz_ratio'])

    return df

