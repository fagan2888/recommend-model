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

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from tabulate import tabulate

import traceback, code

logger = logging.getLogger(__name__)

@click.command()
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
    
    # 加载时间轴数据
    index = DBData.trade_date_index(startdate, end_date=enddate)
    
    db_asset = create_engine(config.db_asset_uri)
    # db_asset.echo = True
    db_base = create_engine(config.db_base_uri)
    db = {'asset':db_asset, 'base':db_base}
    
    metadata = MetaData(bind=db['asset'])
    ra_pool = Table('ra_pool', metadata, autoload=True)

    columns = [
        ra_pool.c.id,
        ra_pool.c.ra_type,
        ra_pool.c.ra_date_type,
        ra_pool.c.ra_fund_type,
        ra_pool.c.ra_lookback,
        ra_pool.c.ra_name,
    ]
    if pools is not None:
        s = select(columns, and_(ra_pool.c.ra_fund_type == 1, ra_pool.c.id.in_(pools.split(','))))
    else:
        s = select(columns, ra_pool.c.ra_fund_type == 1)
    df_pool = pd.read_sql(s, db['asset'])

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_pool, headers='keys', tablefmt='psql')
        return 0
    
    for _, pool in df_pool.iterrows():
        stock_update(db, pool, optlimit, optcalc)
    
def stock_update(db, pool, optlimit, optcalc):
    label_index = pd.DatetimeIndex(['2015-04-03', '2015-09-30', '2016-04-08', '2016-10-14'])

    lookback = pool.ra_lookback
    limit = optlimit

    if optcalc:
        #
        # 计算每个调仓点的最新配置
        #
        data_stock = {}
        for day in label_index:
            data_stock[day] = LabelAsset.label_asset_stock_per_day(day, lookback, limit)

        df_stock = pd.concat(data_stock, names=['ra_date', 'ra_category', 'ra_fund_code'])

        df_new = df_stock.rename(index=DFUtil.categories_types(), columns={'date':'ra_date', 'category':'ra_category', 'code':'ra_fund_code', 'sharpe':'ra_sharpe',  'jensen':'ra_jensen', 'sortino':'ra_sortino', 'ppw':'ra_ppw'})
        df_new.drop('stability', axis=1, inplace=True)

        df_new = df_new.applymap(lambda x: round(x, 4) if type(x) == float else x)
        
        codes = df_new.index.get_level_values(2)
        xtab = fund_code_to_globalid(db['base'], codes)
        df_new['ra_fund_id'] = xtab[df_new.index.get_level_values('ra_fund_code')].values
        df_new['ra_pool'] = pool.id
        df_new['ra_fund_type'] = 1

        df_new.reset_index(inplace=True)
        df_new = df_new.reindex_axis(['ra_pool', 'ra_category',  'ra_date', 'ra_fund_id', 'ra_fund_code', 'ra_fund_type', 'ra_sharpe', 'ra_jensen', 'ra_sortino', 'ra_ppw'], axis='columns')
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
    
def db_batch(db, table, df_new, df_old):
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

            columns['updated_at'] = now

            values = {k:v for k,v in columns.iteritems()}
            stmt = table.update(values=values)
            for k, v in pkeys:
                stmt = stmt.where(table.c.get(k) == v)

            logger.info("update %s : {key: %s, dirties: %s " % (table.name, str(pkeys), str(dirty)))
            stmt.execute()





