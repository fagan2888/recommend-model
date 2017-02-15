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
from db import database, base_ra_fund_nav, base_ra_index, base_ra_fund, base_ra_index_nav

import traceback, code

logger = logging.getLogger(__name__)

@click.group()  
@click.pass_context
def fund(ctx):
    '''fund pool group
    '''
    pass

@fund.command()
@click.option('--id', 'optid', help=u'specify ra corr id (e.g. 500001,500002')
@click.option('--fund', 'optfund', help=u'specify fund code (e.g. 519983,213009')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list corr to update')
@click.pass_context
def corr(ctx, optid, optfund, optlist):
    ''' calc pool corr
    '''
    corrs = None
    if optid is not None:
        corrs = optid.split(',')

    codes = None
    if optfund is not None:
        codes = optfund.split(',')
        
    df_corr = load_ra_corr(corrs)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_corr['ra_name'] = df_corr['ra_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_corr, headers='keys', tablefmt='psql')
        return 0
    
    for _, corr in df_corr.iterrows():
        corr_update(corr, codes)

def corr_update(corr, codes):
    ra_index = base_ra_index.find(corr['ra_index_id'])
    if ra_index is None:
        click.echo(click.style(
            "unknown index [%s]for calc corr!" % (corr['ra_index_id']), fg="yellow"))
        return False

    yesterday = (datetime.now() - timedelta(days=1)); 
    enddate = yesterday.strftime("%Y-%m-%d")        
    
    #
    # 加载指数数据
    #
    index_code = ra_index['ra_code']
    if corr['ra_date_type'] == 1:
        df_nav_index = DBData.db_index_value_daily('2015-10-08', enddate, codes=[index_code])
    else:
        df_nav_index = DBData.db_index_value('2015-10-08', enddate, codes=[index_code])
    df_inc_index = df_nav_index.pct_change().fillna(0.0)

    #
    # 加载基金列表
    #
    df_fund = base_ra_fund.load(codes=codes)
    
    data = []
    with click.progressbar(length=len(df_fund.index), label='update corr for corr %d' % (corr['globalid'])) as bar:
        for _,fund in df_fund.iterrows():
            bar.update(1)
            tmp = corr_update_fund(corr, fund, df_inc_index)
            if tmp is not None:
                data.append([
                    corr['globalid'],
                    fund['globalid'],
                    fund['ra_code'],
                    "%.4f" % (tmp),
                ])

    df_new = pd.DataFrame(data, columns=['ra_corr_id', 'ra_fund_id', 'ra_fund_code', 'ra_corr'])
    df_new = df_new.set_index(['ra_corr_id', 'ra_fund_id'])
    
    db = database.connection('base')
    # 加载旧数据
    t2 = Table('ra_corr_fund', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.ra_corr_id,
        t2.c.ra_fund_id,
        t2.c.ra_fund_code,
        t2.c.ra_corr,
    ]
    stmt_select = select(columns2, (t2.c.ra_corr_id == corr['globalid']))
    if codes is not None:
        stmt_select = stmt_select.where(t2.c.ra_fund_code.in_(codes))
        
    df_old = pd.read_sql(stmt_select, db, index_col=['ra_corr_id', 'ra_fund_id'])
    if not df_old.empty:
        df_old['ra_corr'] = df_old['ra_corr'].map("{:.4f}".format)

    # 更新数据库
    database.batch(db, t2, df_new, df_old, timestamp=True)
            
def corr_update_fund(corr, fund, df_inc_index):
    yesterday = (datetime.now() - timedelta(days=1)); 
    enddate = yesterday.strftime("%Y-%m-%d")        
    #
    # 加载基金数据
    #
    if corr['ra_date_type'] == 1:
        # df_nav_fund = DBData.db_fund_value_daily('2015-10-08', enddate, codes=[fund['ra_code']])
        df_nav_fund = base_ra_fund_nav.load_daily('2015-10-08', enddate, codes=[fund['ra_code']])
    else:
        # df_nav_fund = DBData.db_fund_value('2015-10-08', enddate, codes=[fund['ra_code']])
        df_nav_fund = base_ra_fund_nav.load_weekly('2015-10-08', enddate, codes=[fund['ra_code']])
    if df_nav_fund.empty:
        logger.warn("missing nav for fund [id: %d, code:%s]", fund['globalid'], fund['ra_code'])
        return None
    
    df_inc_fund = df_nav_fund.pct_change().fillna(0.0)
    
    # print df_inc_index.head()
    # print fund, df_inc_fund.head()
    df_inc = pd.DataFrame({'ra_index':df_inc_index.iloc[:, 0], 'ra_fund':df_inc_fund.ix[df_inc_index.index, 0]})
    df_inc.fillna(0.0, inplace=True)
    df_corr = df_inc.corr()
    df_corr.fillna(0.0, inplace=True)

    if df_corr.empty:
        corr = 0.0
    else:
        corr = df_corr.loc['ra_index', 'ra_fund']

    return corr


def load_ra_corr(corrs):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_corr', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.ra_index_id,
        t.c.ra_date_type,
        t.c.ra_lookback,
        t.c.ra_name,
    ]

    s = select(columns)
    if corrs is not None:
        s = s.where(t.c.globalid.in_(corrs))

    df = pd.read_sql(s, db)

    return df
    
@fund.command(name='type')
@click.option('--id', 'optid', help=u'specify type id (e.g. 1001,1002')
@click.option('--fund', 'optfund', help=u'specify fund code (e.g. 519983,213009')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list fund to update')
@click.pass_context
def type_command(ctx, optid, optfund, optlist):
    ''' calc fund type base on corr
    '''
    types = None
    if optid is not None:
        types = optid.split(',')

    codes = None
    if optfund is not None:
        codes = optfund.split(',')
        
    df_corr = load_ra_corr(corrs)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_corr['ra_name'] = df_corr['ra_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_corr, headers='keys', tablefmt='psql')
        return 0
    
    for _, corr in df_corr.iterrows():
        corr_update(corr, codes)
