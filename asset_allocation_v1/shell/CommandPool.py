#coding=utf8


import getopt
import string
import json
import os
import sys
import logging
sys.path.append('shell')
import click
import config
import pandas as pd
import numpy as np
import LabelAsset
import EqualRiskAssetRatio
import EqualRiskAsset
import HighLowRiskAsset
import os
import DBData
import AllocationData
import time
import RiskHighLowRiskAsset
import ModelHighLowRisk
import GeneralizationPosition
import Const
import WeekFund2DayNav
import FixRisk
import DFUtil
import LabelAsset
import database

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from tabulate import tabulate

import traceback, code

logger = logging.getLogger(__name__)

@click.group()  
@click.pass_context
def pool(ctx):
    '''fund pool group
    '''
    pass

@pool.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2010-01-08', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.option('--pools', help=u'fund pool to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.option('--calc/--no-calc', 'optcalc', default=True, help=u're calc label')
@click.option('--limit', 'optlimit', type=int, default=5, help=u'how many fund selected for each category')
@click.pass_context
def stock(ctx, datadir, startdate, enddate, pools, optlist, optlimit, optcalc):
    '''run constant risk model
    '''    
    Const.datadir = datadir

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")        
    
    db_asset = create_engine(config.db_asset_uri)
    # db_asset.echo = True
    db_base = create_engine(config.db_base_uri)
    db = {'asset':db_asset, 'base':db_base}

    df_pool = load_pools(pools, 1)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_pool, headers='keys', tablefmt='psql')
        return 0
    
    for _, pool in df_pool.iterrows():
        stock_update(db, pool, optlimit, optcalc)

def stock_update(db, pool, optlimit, optcalc):
    label_index = get_adjust_point()

    lookback = pool.ra_lookback
    limit = optlimit

    if optcalc:
        #
        # 计算每个调仓点的最新配置
        #
        data_stock = {}
        with click.progressbar(length=len(label_index), label='calc pool %d' % (pool.id)) as bar:
            for day in label_index:
                data_stock[day] = LabelAsset.label_asset_stock_per_day(day, lookback, limit)
                bar.update(1)

        df_stock = pd.concat(data_stock, names=['ra_date', 'ra_category', 'ra_fund_code'])

        df_new = df_stock.rename(index=DFUtil.categories_types(True), columns={'date':'ra_date', 'category':'ra_category', 'code':'ra_fund_code', 'sharpe':'ra_sharpe',  'jensen':'ra_jensen', 'sortino':'ra_sortino', 'ppw':'ra_ppw'})
        df_new.drop('stability', axis=1, inplace=True)

        df_new = df_new.applymap(lambda x: round(x, 4) if type(x) == float else x)
        
        codes = df_new.index.get_level_values(2)
        xtab = fund_code_to_globalid(db['base'], codes)
        df_new['ra_fund_id'] = xtab[df_new.index.get_level_values('ra_fund_code')].values
        df_new['ra_pool'] = pool.id
        df_new['ra_fund_type'] = 1
        df_new['ra_fund_level'] = 1

        df_new.reset_index(inplace=True)
        df_new = df_new.reindex_axis(['ra_pool', 'ra_category',  'ra_date', 'ra_fund_id', 'ra_fund_code', 'ra_fund_type', 'ra_fund_level', 'ra_sharpe', 'ra_jensen', 'ra_sortino', 'ra_ppw'], axis='columns')
        df_new.sort_values(by=['ra_pool', 'ra_category', 'ra_date', 'ra_fund_id'], inplace=True)
        
        df_new.to_csv(datapath('pool_%d.csv' % (pool['id'])), index=False)
        
    else:
        df_new = pd.read_csv(datapath('pool_%d.csv' % (pool['id'])), parse_dates=['ra_date'], dtype={'ra_fund_code': str})
        
    df_new.set_index(['ra_pool', 'ra_category', 'ra_date', 'ra_fund_id'], inplace=True)
    df_new = df_new.applymap(lambda x: '%.4f' % (x) if type(x) == float else x)

    ra_pool_fund = Table('ra_pool_fund', MetaData(bind=db['asset']), autoload=True)

    # 加载就数据
    columns2 = [
        ra_pool_fund.c.ra_pool,
        ra_pool_fund.c.ra_category,
        ra_pool_fund.c.ra_date,
        ra_pool_fund.c.ra_fund_id,
        ra_pool_fund.c.ra_fund_code,
        ra_pool_fund.c.ra_fund_type,
        ra_pool_fund.c.ra_fund_level,
        ra_pool_fund.c.ra_sharpe,
        ra_pool_fund.c.ra_jensen,
        ra_pool_fund.c.ra_sortino,
        ra_pool_fund.c.ra_ppw,
    ]
    stmt_select = select(columns2, ra_pool_fund.c.ra_pool == pool.id)
    df_old = pd.read_sql(stmt_select, db['asset'], index_col=['ra_pool', 'ra_category', 'ra_date', 'ra_fund_id'])
    if not df_old.empty:
        df_old = df_old.applymap(lambda x: '%.4f' % (x) if type(x) == float else x)

    db_batch(db['asset'], ra_pool_fund, df_new, df_old)

def fund_code_to_globalid(db, codes):
    metadata = MetaData(bind=db)
    ra_fund = Table('ra_fund', metadata, autoload=True)

    columns = [
        ra_fund.c.globalid,
        ra_fund.c.ra_code,
    ]
    if codes is not None:
        s = select(columns, ra_fund.c.ra_code.in_(codes))
    else:
        s = select(columns)
        
    df_result = pd.read_sql(s, db, index_col='ra_code')
    
    return df_result['globalid']
    
def db_batch(db, table, df_new, df_old, timestamp=True):
    now = datetime.now()
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
            df_insert['updated_at'] = df_insert['created_at'] = now

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
                columns['updated_at'] = now

            values = {k:v for k,v in columns.iteritems()}
            stmt = table.update(values=values)
            for k, v in pkeys:
                stmt = stmt.where(table.c.get(k) == v)

            logger.info("update %s : {key: %s, dirties: %s " % (table.name, str(pkeys), str(dirty)))
            stmt.execute()


@pool.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2010-01-08', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.option('--pools', help=u'fund pool to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.option('--calc/--no-calc', 'optcalc', default=True, help=u're calc label')
@click.option('--limit', 'optlimit', type=int, default=5, help=u'how many fund selected for each category')
@click.pass_context
def bond(ctx, datadir, startdate, enddate, pools, optlist, optlimit, optcalc):
    '''run constant risk model
    '''    
    Const.datadir = datadir

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")        
    
    # 加载时间轴数据
    index = DBData.trade_date_index(startdate, end_date=enddate)
    
    db_asset = create_engine(config.db_asset_uri)
    # db_asset.echo = True
    db_base = create_engine(config.db_base_uri)
    db = {'asset':db_asset, 'base':db_base}

    df_pool = load_pools(pools, 2)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_pool, headers='keys', tablefmt='psql')
        return 0
    
    for _, pool in df_pool.iterrows():
        bond_update(db, pool, optlimit, optcalc)

def bond_update(db, pool, optlimit, optcalc):
    label_index = get_adjust_point()

    lookback = pool.ra_lookback
    limit = optlimit

    if optcalc:
        #
        # 计算每个调仓点的最新配置
        #
        data_bond = {}
        with click.progressbar(length=len(label_index), label='calc pool %d' % (pool.id)) as bar:
            for day in label_index:
                data_bond[day] = LabelAsset.label_asset_bond_per_day(day, lookback, limit)
                bar.update(1)

        df_bond = pd.concat(data_bond, names=['ra_date', 'ra_category', 'ra_fund_code'])

        df_new = df_bond.rename(index=DFUtil.categories_types(True), columns={'date':'ra_date', 'category':'ra_category', 'code':'ra_fund_code', 'sharpe':'ra_sharpe',  'jensen':'ra_jensen', 'sortino':'ra_sortino', 'ppw':'ra_ppw'})
        df_new.drop('stability', axis=1, inplace=True)

        df_new = df_new.applymap(lambda x: round(x, 4) if type(x) == float else x)
        
        codes = df_new.index.get_level_values(2)
        xtab = fund_code_to_globalid(db['base'], codes)
        df_new['ra_fund_id'] = xtab[df_new.index.get_level_values('ra_fund_code')].values
        df_new['ra_pool'] = pool.id
        df_new['ra_fund_type'] = 1
        df_new['ra_fund_level'] = 1

        df_new.reset_index(inplace=True)
        df_new = df_new.reindex_axis(['ra_pool', 'ra_category',  'ra_date', 'ra_fund_id', 'ra_fund_code', 'ra_fund_type', 'ra_fund_level', 'ra_sharpe', 'ra_jensen', 'ra_sortino', 'ra_ppw'], axis='columns')
        df_new.sort_values(by=['ra_pool', 'ra_category', 'ra_date', 'ra_fund_id'], inplace=True)
        
        df_new.to_csv(datapath('pool_%d.csv' % (pool['id'])), index=False)
        
    else:
        df_new = pd.read_csv(datapath('pool_%d.csv' % (pool['id'])), parse_dates=['ra_date'], dtype={'ra_fund_code': str})
        
    df_new.set_index(['ra_pool', 'ra_category', 'ra_date', 'ra_fund_id'], inplace=True)
    df_new = df_new.applymap(lambda x: '%.4f' % (x) if type(x) == float else x)

    ra_pool_fund = Table('ra_pool_fund', MetaData(bind=db['asset']), autoload=True)

    # 加载就数据
    columns2 = [
        ra_pool_fund.c.ra_pool,
        ra_pool_fund.c.ra_category,
        ra_pool_fund.c.ra_date,
        ra_pool_fund.c.ra_fund_id,
        ra_pool_fund.c.ra_fund_code,
        ra_pool_fund.c.ra_fund_type,
        ra_pool_fund.c.ra_fund_level,
        ra_pool_fund.c.ra_sharpe,
        ra_pool_fund.c.ra_jensen,
        ra_pool_fund.c.ra_sortino,
        ra_pool_fund.c.ra_ppw,
    ]
    stmt_select = select(columns2, ra_pool_fund.c.ra_pool == pool.id)
    df_old = pd.read_sql(stmt_select, db['asset'], index_col=['ra_pool', 'ra_category', 'ra_date', 'ra_fund_id'])
    if not df_old.empty:
        df_old = df_old.applymap(lambda x: '%.4f' % (x) if type(x) == float else x)

    db_batch(db['asset'], ra_pool_fund, df_new, df_old)


def load_pools(pools, pool_type=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    ra_pool = Table('ra_pool', metadata, autoload=True)

    columns = [
        ra_pool.c.id,
        ra_pool.c.ra_type,
        ra_pool.c.ra_date_type,
        ra_pool.c.ra_fund_type,
        ra_pool.c.ra_lookback,
        ra_pool.c.ra_name,
    ]

    s = select(columns)
    if pools is not None:
        s = s.where(ra_pool.c.id.in_(pools.split(',')))
    if pool_type is not None:
        s = s.where(ra_pool.c.ra_fund_type == pool_type)
        
    df_pool = pd.read_sql(s, db)

    return df_pool
        
def get_adjust_point(startdate = '2010-01-08', enddate=None, label_period=13):
    # 加载时间轴数据
    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")        

    index = DBData.trade_date_index(startdate, end_date=enddate)
    
    if label_period > 1:
        label_index = index[::label_period]
        if index.max() not in label_index:
            label_index = label_index.insert(len(label_index), index.max())
    else:
        label_index = index
    # label_index = pd.DatetimeIndex([
    #     '2010-01-08',
    #     '2010-07-09',
    #     '2011-01-07',
    #     '2011-07-08',
    #     '2012-01-13',
    #     '2012-07-20',
    #     '2013-01-25',
    #     '2013-08-02',
    #     '2014-01-30',
    #     '2014-08-01',
    #     '2015-01-30',
    #     '2015-07-31',
    #     '2016-01-29',
    #     '2016-08-05',
    #     '2016-11-05',
    # ])
    
    print "adjust point:"
    for date in label_index:
        print date.strftime("%Y-%m-%d")

    return label_index

@pool.command()
@click.option('--pools', help=u'fund pool to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.pass_context
def nav(ctx, pools, optlist):
    ''' calc pool nav and inc
    '''
    db_asset = create_engine(config.db_asset_uri)
    # db_asset.echo = True
    db_base = create_engine(config.db_base_uri)
    db = {'asset':db_asset, 'base':db_base}

    df_pool = load_pools(pools)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_pool, headers='keys', tablefmt='psql')
        return 0
    
    for _, pool in df_pool.iterrows():
        nav_update(db, pool)

def nav_update(db, pool):
    df_categories = load_pool_category(pool['id'])
    categories = df_categories['ra_category']
    
    with click.progressbar(length=len(categories) + 1, label='update nav for pool %d' % (pool.id)) as bar:
        for category in categories:
            nav_update_category(db['asset'], pool, category)
            bar.update(1)
        else:
            nav_update_category(db['asset'], pool, 0)
            bar.update(1)
    #nav_update_category(db['asset'], pool, 0)

def nav_update_category(db, pool, category):
    # 加载基金列表
    df = load_fund_category(pool['id'], category)

    # 构建均分仓位
    df['ra_ratio'] = 1.0
    df.set_index('ra_fund_code', append=True, inplace=True)
    df['ra_ratio'] = df['ra_ratio'].groupby(level=0, group_keys=False).apply(lambda x: x / len(x))
    df_position = df.unstack().fillna(0.0)
    df_position.columns = df_position.columns.droplevel(0)

    
    # 加载基金收益率
    min_date = df_position.index.min()
    #max_date = df_position.index.max()
    max_date = (datetime.now() - timedelta(days=1)) # yesterday


    df_nav = DBData.db_fund_value_daily(
        min_date, max_date, codes=df_position.columns)
    df_inc = df_nav.pct_change().fillna(0.0)

    # 计算复合资产净值
    df_nav_portfolio = DFUtil.portfolio_nav(df_inc, df_position, result_col='portfolio')
    # df_nav_portfolio.to_csv(datapath('category_nav_' + category + '.csv'))

    df_result = df_nav_portfolio[['portfolio']].rename(columns={'portfolio':'ra_nav'}).copy()
    df_result.index.name = 'ra_date'
    df_result['ra_inc'] = df_result['ra_nav'].pct_change().fillna(0.0)
    df_result['ra_pool'] = pool['id']
    df_result['ra_category'] = category
    df_result['ra_type'] = pool['ra_type']
    df_result = df_result.reset_index().set_index(['ra_pool', 'ra_category', 'ra_type', 'ra_date'])
    
    df_new = df_result.apply(format_nav_and_inc)


    # 加载旧数据
    t2 = Table('ra_pool_nav', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.ra_pool,
        t2.c.ra_category,
        t2.c.ra_type,
        t2.c.ra_date,
        t2.c.ra_nav,
        t2.c.ra_inc,
    ]
    stmt_select = select(columns2, (t2.c.ra_pool == pool['id']) & (t2.c.ra_category == category) & (t2.c.ra_type == pool['ra_type']))
    df_old = pd.read_sql(stmt_select, db, index_col=['ra_pool', 'ra_category', 'ra_type', 'ra_date'], parse_dates=['ra_date'])
    if not df_old.empty:
        df_old = df_old.apply(format_nav_and_inc)

    # 更新数据库
    db_batch(db, t2, df_new, df_old, timestamp=False)

def format_nav_and_inc(x):
    if x.name == "ra_nav":
        ret = x.map("{:.6f}".format)
    elif x.name == "ra_inc":
        ret = x.map("{:.6f}".format)
    else:
        ret = x

    return ret
    
    
def load_pool_category(pid):
    db = database.connection('asset')
    t = Table('ra_pool_fund', MetaData(bind=db), autoload=True)
    
    # 加载就数据
    columns = [
        t.c.ra_pool,
        t.c.ra_category,
    ]
    stmt_select = select(columns, t.c.ra_pool == pid).distinct()
    
    df = pd.read_sql(stmt_select, db)

    return df

def load_fund_category(pid, category):
    db = database.connection('asset')
    # 加载基金列表
    t = Table('ra_pool_fund', MetaData(bind=db), autoload=True)
    columns = [
        t.c.ra_date,
        t.c.ra_fund_code,
    ]
    s = select(columns, (t.c.ra_pool == pid))
    if category == 0:
        s = s.distinct()
    else:
        s = s.where(t.c.ra_category == category)
    
    df = pd.read_sql(s, db, index_col = ['ra_date'], parse_dates=['ra_date'])

    return df
    
    
@pool.command(name='import')
@click.option('--id', 'optid', type=int, help=u'specify fund pool id')
@click.option('--name', 'optname', type=int, help=u'specify fund pool name')
@click.option('--otype', 'optotype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--dtype', 'optdtype', type=click.Choice(['1', '2']), default='2', help=u'date type(1:day, 2:week)')
@click.option('--ftype', 'optftype', type=click.Choice(['0', '1', '2', '3', '4']), default='0', help=u'fund type(0:unknow; 1:stock; 2:bond; 3:money; 4:other)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace pool if exists')
@click.argument('csv', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=False), required=True)
@click.pass_context
def import_command(ctx, csv, optid, optname, optotype, optdtype, optftype, optreplace):
    '''
    import fund pool from csv file
    '''

    #
    # 处理id参数
    #
    if optid is not None:
        #
        # 检查id是否存在
        #
        df_existed = load_pools(str(optid))
        if not df_existed.empty:
            if optreplace:
                click.echo(click.style("allocation instance [optId] existed, will replace!", fg="yellow"))
            else:
                click.echo(click.style("allocation instance [optId] existed, import aborted!", fg="red"))
            return -1;
    else:
        #
        # 自动生成id
        #
        prefix = optotype + optdtype + optftype
        between_min, between_max = ('%s00' % (prefix), '%s99' % (prefix))

        max_id = query_max_pool_id_between(between_min, between_max)
        if max_id is None:
            optid = between_min;
        else:
            if max_id >= between_max:
                click.echo(click.style("run out of instance id [maxId], import aborted!", fg="red"))
                return -1

            if optreplace:
                optid = max_id
            else:
                optid = max_id + 1;

    #
    # 处理name参数
    #
    if optname is None:
        optname = os.path.basename(csv);


    db = database.connection('asset')
    metadata = MetaData(bind=db)
    ra_pool = Table('ra_pool', metadata, autoload=True)
    ra_pool_fund = Table('ra_pool_fund', metadata, autoload=True)

    #
    # 处理替换
    #
    if optreplace:
        ra_pool.delete(ra_pool.c.id == optid).execute()
        ra_pool_fund.delete(ra_pool_fund.c.ra_pool_id == optid).execute()

    now = datetime.now()
    #
    # 导入数据
    #
    row = {
        'id': optid, 'ra_type':optotype, 'ra_date_type':optdtype
        , 'ra_fund_type':optftype, 'ra_name': optname
        , 'created_at': func.now(), 'updated_at': func.now()
    }
    ra_pool.insert(row).execute()

    df = pd.read_csv(csv, parse_dates=['date'], dtype={'code':str})
    df = df.replace({'category':DFUtil.categories_types()})
    df = df.rename(columns={'date':'ra_date', 'category':'ra_category', 'code':'ra_fund_code', 'sharpe':'ra_sharpe',  'jensen':'ra_jensen', 'sortino':'ra_sortino', 'ppw':'ra_ppw', 'level':'ra_fund_level'})

    if 'ra_fund_level' not in df.columns:
        df['ra_fund_level'] = 1

    if 'ra_jensen' not in df.columns:
        df['ra_jensen'] = 0.0
    else:
        df['ra_jensen'] = np.round(df['ra_jensen'])

    if 'ra_sharpe' not in df.columns:
        df['ra_sharpe'] = 0.0
    else:
        df['ra_sharpe'] = np.round(df['ra_sharpe'])

    if 'ra_ppw' not in df.columns:
        df['ra_ppw'] = 0.0
    else:
        df['ra_ppw'] = np.round(df['ra_ppw'])

    if 'ra_sortino' not in df.columns:
        df['ra_sortino'] = 0.0
    else:
        df['ra_sortino'] = np.round(df['ra_sortino'])

    if 'stability' in df.columns:
        df =  df.drop('stability', axis = 1)


    #
    # 翻译基金代码 => 基金ID
    #
    df_tosave = DFUtil.merge_column_for_fund_id_type(df, 'ra_fund_code')
    df_tosave = df_tosave.rename(columns={'globalid':'ra_fund_id', 'ra_type':'ra_fund_type'})
    df_tosave['ra_pool'] = optid
    df_tosave.set_index(['ra_pool', 'ra_category', 'ra_date', 'ra_fund_id'], inplace=True)

    df_tosave['updated_at'] = df_tosave['created_at'] = now

    df_tosave.to_sql(ra_pool_fund.name, db, index=True, if_exists='append')

    if len(df_tosave.index) > 1:
        logger.info("insert %s (%5d) : %s " % (ra_pool_fund.name, len(df_tosave.index), df_tosave.index[0]))

    click.echo(click.style("import complement! instance id [%s]" % (optid), fg='green'))
    
    
    return 0

def query_max_pool_id_between(min_id, max_id):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('ra_pool', metadata, autoload=True)

    columns = [ t.c.id ]

    s = select([func.max(t.c.id).label('maxid')]).where(t.c.id.between(min_id, max_id))

    return s.execute().scalar()

@pool.command()
@click.option('--id', 'optid', help=u'specify fund pool id (e.g. 12101,92101')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.pass_context
def turnover(ctx, optid, optlist):
    ''' calc pool turnover
    '''
    df_pool = load_pools(optid)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_pool, headers='keys', tablefmt='psql')
        return 0
    
    for _, pool in df_pool.iterrows():
        turnover_update(pool)

def turnover_update(pool):
    df_categories = load_pool_category(pool['id'])
    categories = df_categories['ra_category']
    
    with click.progressbar(length=len(categories) + 1, label='update turnover for pool %d' % (pool.id)) as bar:
        for category in categories:
            turnover_update_category(pool, category)
            bar.update(1)
        else:
            turnover_update_category(pool, 0)
            bar.update(1)

def turnover_update_category(pool, category):
    # 加载基金列表
    df = load_fund_category(pool['id'], category)

    # 构建均分仓位
    df['mask'] = 1
    df.set_index('ra_fund_code', append=True, inplace=True)
    df = df.unstack(fill_value=0)
    df_prev = df.shift(1).fillna(0).astype(int)

    df_prev.iloc[0] = df.iloc[0]
    
    df_and = np.bitwise_and(df, df_prev)

    df_new = (1 - df_and.sum(axis=1) / df_prev.sum(axis=1)).to_frame('ra_turnover')
    df_new['ra_pool'] = pool['id']
    df_new['ra_category'] = category

    df_new = df_new.reset_index().set_index(['ra_pool', 'ra_category', 'ra_date'])
    
    df_new = df_new.applymap("{:.4f}".format)


    db = database.connection('asset')
    # 加载旧数据
    t2 = Table('ra_pool_criteria', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.ra_pool,
        t2.c.ra_category,
        t2.c.ra_date,
        t2.c.ra_turnover,
    ]
    stmt_select = select(columns2, (t2.c.ra_pool == pool['id']) & (t2.c.ra_category == category))
    df_old = pd.read_sql(stmt_select, db, index_col=['ra_pool', 'ra_category', 'ra_date'], parse_dates=['ra_date'])
    if not df_old.empty:
        df_old = df_old.applymap("{:.4f}".format)

    # 更新数据库
    db_batch(db, t2, df_new, df_old, timestamp=False)
    
