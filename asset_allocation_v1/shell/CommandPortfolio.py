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
from db import database

import traceback, code

logger = logging.getLogger(__name__)

@click.group()  
@click.pass_context
def portfolio(ctx):
    ''' generate portfolios
    '''
    pass;
    
@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--output', '-o', type=click.File(mode='w'), default='-', help=u'file used to store final result')
# @click.option('-m', '--msg')  
# @click.option('--dry-run', is_flag=True, help=u'pretend to run')
# @click.option('--name', prompt='Your name', help='The person to greet.')
@click.pass_context
def simple(ctx, datadir, output):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    out = output
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    all_code_position = GeneralizationPosition.risk_position()
    
    GeneralizationPosition.output_final_portfolio(all_code_position, out)
    
@portfolio.command()  
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--output', '-o', type=click.File(mode='w'), default='-', help=u'file used to store final result')
@click.pass_context
def optimize(ctx, datadir, output):
    '''generate final portfolio with optimized strategy (cost consider in).  
    '''
    out = output
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    all_code_position = GeneralizationPosition.risk_position()
    
    GeneralizationPosition.output_portfolio(all_code_position, out)

@portfolio.command()  
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--output', '-o', type=click.File(mode='w'), default='-', help=u'file used to store final result')
@click.pass_context
def category(ctx, datadir, output):
    '''generate intemediate portfolio for different asset categories 
    '''
    out = output
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    all_code_position = GeneralizationPosition.risk_position()
    
    GeneralizationPosition.output_category_portfolio(all_code_position, out)

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
# @click.option('-m', '--msg')  
# @click.option('--dry-run', is_flag=True, help=u'pretend to run')
# @click.option('--name', prompt='Your name', help='The person to greet.')
@click.pass_context
def ncat(ctx, datadir):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    GeneralizationPosition.portfolio_category()

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
# @click.option('-m', '--msg')  
# @click.option('--dry-run', is_flag=True, help=u'pretend to run')
# @click.option('--name', prompt='Your name', help='The person to greet.')
@click.pass_context
def nsimple(ctx, datadir):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    GeneralizationPosition.portfolio_simple()

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.pass_context
def detail(ctx, datadir):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    GeneralizationPosition.portfolio_detail()

@portfolio.command()  
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--output', '-o', default=None, help=u'file used to store final result')
@click.pass_context
def trade(ctx, datadir, output):
    '''generate final portfolio with optimized strategy (cost consider in).  
    '''
    Const.datadir = datadir
    if output is None:
        output = datapath('position-z.csv')
    with (open(output, 'w') if output != '-' else os.fdopen(os.dup(sys.stdout.fileno()), 'w')) as out:
        GeneralizationPosition.portfolio_trade(out)

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--input', '-i', 'optinput', type=click.Path(exists=True), help=u'portfolio position file as input') 
@click.option('--output', '-o', 'optoutput', type=click.Path(), help=u'file position file to output') 
@click.pass_context
def stockavg(ctx, datadir, optinput, optoutput):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    output = None
    if optinput is None:
        if os.path.isfile(datapath('riskmgr_position.csv')):
            optinput = datapath('riskmgr_position.csv')
            output = datapath('position-r.csv')
        elif os.path.isfile(datapath('portfolio_position.csv')):
            optinput = datapath('portfolio_position.csv')
            output = datapath('position-v.csv')
        else:
            click.echo(click.style("error: mising position file!", fg="yellow"))
            return -1
        
    if optoutput is None:
        if output is None:
            optoutput = datapath('position-v.csv')
        else:
            optoutput = output
                
        
    print "convert portfilio position  %s to final position %s" % (optinput, optoutput)
    GeneralizationPosition.portfolio_avg_simple(optinput, optoutput)

@portfolio.command()
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
    s = select(columns, and_(t.c.ai_inst_id == inst, t.c.ai_alloc_id == alloc)) \
        .order_by(t.c.ai_transfer_date, t.c.ai_category)    
    df = pd.read_sql(s, db, parse_dates=['ai_transfer_date'])

    return df

@portfolio.command()
@click.option('--inst', 'optInst', help=u'portfolio id to calc turnover')
@click.option('--alloc', 'optAlloc', help=u'risk of portfolio to calc turnover')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.option('--start-date', 'startdate', default='2010-01-08', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.pass_context
def export(ctx, optInst, optAlloc, startdate, enddate, optlist):
    '''run constant risk model
    '''    
    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")        
    
    db_asset = create_engine(config.db_asset_uri)
    # db_asset.echo = True
    # db_base = create_engine(config.db_base_uri)
    db = {'asset':db_asset}

    df = load_portfolio_category_by_id(db['asset'], optInst, optAlloc)
    df = df.set_index(['ai_alloc_id', 'ai_transfer_date',  'ai_category'])

    df_result = df.unstack().fillna(0.0)
    df_result.columns = df_result.columns.droplevel(0)

    path = datapath('%s-%s-category.csv' % (optInst, optAlloc))
    df_result.to_csv(path)

    print "export allocation instance [%s(risk:%s)] to file %s" % (optInst, optAlloc, path)


def load_portfolio_category_by_id(db, inst, alloc):
    metadata = MetaData(bind=db)
    t = Table('allocation_instance_position_detail', metadata, autoload=True)

    columns = [
        t.c.ai_alloc_id,
        t.c.ai_transfer_date,
        t.c.ai_category,
        func.sum(t.c.ai_fund_ratio).label('ratio')
    ]
    s = select(columns, and_(t.c.ai_inst_id == inst, t.c.ai_alloc_id == alloc)) \
        .group_by(t.c.ai_transfer_date, t.c.ai_category) \
        .order_by(t.c.ai_transfer_date, t.c.ai_category)    
    df = pd.read_sql(s, db, parse_dates=['ai_transfer_date'])

    return df


@portfolio.command()
@click.option('--from', 'optfrom', help=u'portfolio id to convert from')
@click.option('--to', 'optto', help=u'portfolio id to convert to')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.pass_context
def convert1(ctx, optfrom, optto, optlist):
    '''convert bond and money to bank
    '''    

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('allocation_instance_position_detail', metadata, autoload=True)
    t2 = Table('allocation_instance_position', metadata, autoload=True)

    columns = [
        t1.c.ai_inst_type,
        t1.c.ai_alloc_id,
        t1.c.ai_transfer_date,
        t1.c.ai_category,
        t1.c.ai_fund_id,
        t1.c.ai_fund_code,
        t1.c.ai_fund_ratio,
    ]

    s = select(columns).where(t1.c.ai_inst_id == optfrom)
        
    df = pd.read_sql(s, db, index_col = ['ai_alloc_id', 'ai_transfer_date'], parse_dates=['ai_transfer_date'])

    mask = (df['ai_category'] >= 20) & (df['ai_category'] < 30)
    df.loc[mask, 'ai_fund_id'] =  33056942
    df.loc[mask, 'ai_fund_code'] = '999002'
    #df.loc[maks, 'ai_category'] = 51

    mask = (df['ai_category'] >= 30) & (df['ai_category'] < 40)
    df.loc[mask, 'ai_fund_id'] =  33056941
    df.loc[mask, 'ai_fund_code'] = '999001'
    #df.loc[maks, 'ai_category'] = 52

    df.set_index(['ai_category', 'ai_fund_id'], append=True, inplace=True)
    df_result = df.groupby(level=(0, 1, 2, 3)).agg({'ai_inst_type':'first', 'ai_fund_code':'first', 'ai_fund_ratio':'sum'})

    t1.delete(t1.c.ai_inst_id == optto).execute()

    df_result['ai_inst_id'] = optto
    df_result['updated_at'] = df_result['created_at'] = datetime.now()

    df_result = df_result.reset_index().set_index(['ai_inst_id', 'ai_alloc_id', 'ai_transfer_date', 'ai_category', 'ai_fund_id'])
    
    df_result.to_sql(t1.name, db, index=True, if_exists='append', flavor='mysql', chunksize=500)


    df_result = df_result.reset_index().set_index(['ai_inst_id', 'ai_alloc_id', 'ai_transfer_date', 'ai_fund_id'])
    df_result['ai_fund_type'] = df_result['ai_category'].floordiv(10)
    df_result =  df_result.drop('ai_category', axis = 1)
    
    df_result = df_result.groupby(level=(0, 1, 2, 3)).agg({
        'ai_inst_type':'first', 'ai_fund_code':'first', 'ai_fund_type':'first', 'ai_fund_ratio':'sum', 'updated_at':'first', 'created_at':'first'
    })
    print df_result.head()

    t2.delete(t2.c.ai_inst_id == optto).execute()
    df_result.to_sql(t2.name, db, index=True, if_exists='append', flavor='mysql', chunksize=500)
    


    
