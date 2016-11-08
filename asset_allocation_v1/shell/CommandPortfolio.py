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
@click.option('--inst', 'optInst', help=u'portfolio id to calc turnover')
@click.option('--alloc', 'optAlloc', help=u'risk of portfolio to calc turnover')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.option('--start-date', 'startdate', default='2010-01-08', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.pass_context
def turnover(ctx, optInst, optAlloc, startdate, enddate, optlist):
    '''run constant risk model
    '''    
    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")        
    
    db_asset = create_engine(config.db_asset_uri)
    # db_asset.echo = True
    # db_base = create_engine(config.db_base_uri)
    db = {'asset':db_asset}

    df = load_portfolio_by_id(db['asset'], optInst, optAlloc)

    df_result = calc_turnover(df)

    total_turnover = df_result.sum()

    df_result.reset_index(inplace=True)
    df_result['turnover'] = df_result['turnover'].map(lambda x: "%6.2f%%" % (x * 100))

    print tabulate(df_result, headers='keys', tablefmt='psql', stralign=u'right')

    print "total turnover: %.2f%%" % (total_turnover * 100)
    
    # if optlist:
    #     #print df_pool
    #     #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
    #     df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
    #     print tabulate(df_pool, headers='keys', tablefmt='psql')
    #     return 0
    
    # for _, pool in df_pool.iterrows():
    #     stock_update(db, pool, optlimit, optcalc)

def calc_turnover(df):

    df.loc[(df['ai_category'] >= 10) & (df['ai_category'] < 30), 'ai_category'] = (df['ai_category'] // 10)
    df.set_index(['ai_transfer_date', 'ai_category', 'ai_fund_code'], inplace=True)
    #df2 = df.groupby(level=(0,1)).agg({'ai_inst_type':'first', 'ai_alloc_id':'first', 'ai_fund_ratio':'sum'})
    df2 = df[['ai_fund_ratio']].groupby(level=(0,1)).sum()
    df2 = df2.unstack().fillna(0.0)
    df2.columns = df2.columns.droplevel(0)

    df_result = df2.rolling(window=2, min_periods=1).apply(lambda x: x[1] - x[0] if len(x) > 1 else x[0])

    df_result = df_result.abs().sum(axis=1).to_frame('turnover')

    return df_result


def load_portfolio_by_id(db, inst, alloc):
    metadata = MetaData(bind=db)
    t = Table('allocation_instance_position_detail', metadata, autoload=True)

    columns = [
        t.c.ai_inst_type,
        t.c.ai_alloc_id,
        t.c.ai_transfer_date,
        t.c.ai_category,
        t.c.ai_fund_code,
        t.c.ai_fund_ratio,
    ]
    s = select(columns, and_(t.c.ai_inst_id == inst, t.c.ai_alloc_id == alloc))
    s.order_by(t.c.ai_transfer_date, t.c.ai_category)    
    df = pd.read_sql(s, db, parse_dates=['ai_transfer_date'])

    return df
        
