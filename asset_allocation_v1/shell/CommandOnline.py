#coding=utf8

import pdb
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
import os
import time
import DFUtil
import util_numpy as npu
import TradeNav

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from tabulate import tabulate
from db import *
from util.xdebug import dd

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)  
@click.option('--id', 'optid', help=u'specify online id')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.pass_context
def online(ctx, optid, opttype):

    '''generate final portolio
    '''
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        ctx.invoke(nav, optid=optid)
        ctx.invoke(turnover, optid=optid)
    else:
        # click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        pass

@online.command()
@click.option('--id', 'optid', help=u'ids of online to update')
@click.option('--fee', 'optfee', default='9', help=u'fee type(8:with fee; 9:without fee')
@click.option('--debug/--no-debug', 'optdebug', default=False, help=u'debug mode')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.pass_context
def nav(ctx, optid, optlist, optfee, optdebug):
    ''' calc pool nav and inc
    '''
    if optid is not None:
        onlines = [s.strip() for s in optid.split(',')]
    else:
        if 'online' in ctx.obj:
            onlines = [str(ctx.obj['online'])]
        else:
            onlines = None

    fees = [int(s.strip()) for s in optfee.split(',')]

    df_online = asset_on_online.load(onlines)

    if optlist:
        df_online['on_name'] = df_online['on_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_online, headers='keys', tablefmt='psql')
        return 0

    for fee in fees:
        for _, online in df_online.iterrows():
            nav_update_alloc(online, fee, optdebug)

def nav_update_alloc(online, fee, debug):
    df_alloc = asset_on_online_alloc.where_online_id(online['globalid'])
    
    with click.progressbar(
            df_alloc.iterrows(), length=len(df_alloc.index),
            label='update nav %-9d' % (online['globalid']),
            item_show_func=lambda x: str(x[1]['globalid']) if x else None) as bar:
        for _, alloc in bar:
    # with click.progressbar(length=len(df_alloc), label='update nav %d' % (online['globalid'])) as bar:
    #     for _, alloc in :
    #         bar.update(1)
            nav_update(alloc, fee, debug)
    
def nav_update(alloc, fee, debug):
    alloc_id = alloc['globalid']
    # 加载仓位信息
    df_pos = asset_on_online_fund.load_fund_pos(alloc_id)
    df_pos.index.names=['ra_date', 'ra_fund_id']
    df_pos = df_pos.rename(columns={'on_fund_ratio': 'ra_fund_ratio'})
    
    max_date = (datetime.now() - timedelta(days=1)) # yesterday

    # 计算复合资产净值
    if fee == 8:
        xtype = 8
        df_pos = df_pos.loc[df_pos.index.get_level_values(0) >= '2012-07-27']
        tn = TradeNav.TradeNav(debug=debug)
        tn.calc(df_pos, 1)
        sr_nav_online = pd.Series(tn.nav)
        sr_contrib = pd.concat(tn.contrib)
    else:
        xtype = 9
        sr_nav_online = DFUtil.online_nav2(df_pos, end_date=max_date)
        sr_contrib = pd.Series()

    df_result = sr_nav_online.to_frame('on_nav')
    df_result.index.name = 'on_date'
    df_result['on_type'] = xtype
    df_result['on_inc'] = df_result['on_nav'].pct_change().fillna(0.0)
    df_result['on_online_id'] = alloc_id
    df_result = df_result.reset_index().set_index(['on_online_id', 'on_type', 'on_date'])

    asset_on_online_nav.save(alloc_id, xtype, df_result)

    if not sr_contrib.empty:
        df_contrib = sr_contrib.to_frame('on_return_value')
        df_contrib.index.names=[u'on_date', u'on_fund_id']
        df_contrib['on_type'] = xtype
        df_contrib['on_online_id'] = alloc_id
        df_contrib['on_return_type'] = 0
        df_contrib = df_contrib.reset_index().set_index(['on_online_id', 'on_type', 'on_date', 'on_fund_id', 'on_return_type'])

        asset_on_online_contrib.save(alloc_id, xtype, df_contrib)
    dd("abort")

@online.command()
@click.option('--id', 'optid', help=u'ids of portfolio to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.pass_context
def turnover(ctx, optid, optlist):
    ''' calc pool turnover and inc
    '''
    if optid is not None:
        onlines = [s.strip() for s in optid.split(',')]
    else:
        if 'online' in ctx.obj:
            onlines = [str(ctx.obj['online'])]
        else:
            onlines = None

    df_online = asset_on_online.load(onlines)

    if optlist:

        df_online['on_name'] = df_online['on_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_online, headers='keys', tablefmt='psql')
        return 0
    
    data = []
    for _, online in df_online.iterrows():
        turnover_update_alloc(online)

def turnover_update_alloc(online):
    df_alloc = asset_on_online_alloc.where_online_id(online['globalid'])
    
    with click.progressbar(
            df_alloc.iterrows(), length=len(df_alloc.index),
            label='turnover %-11d' % (online['globalid']),
            item_show_func=lambda x:  str(x[1]['globalid']) if x else None) as bar:
        for _, alloc in bar:
    # with click.progressbar(length=len(df_alloc), label='update turnover %d' % (online['globalid'])) as bar:
    #     for _, alloc in df_alloc.iterrows():
    #         bar.update(1)
            turnover_update(alloc)

            
def turnover_update(online):
    online_id = online['globalid']
    # 加载仓位信息
    df = asset_on_online_pos.load_fund_pos(online_id)
    df = df.unstack()


    # 计算宽口换手率
    sr_turnover = DFUtil.calc_turnover(df)

    criteria_id = 6
    df_result = sr_turnover.to_frame('on_value')
    df_result['on_online_id'] = online_id
    df_result['on_criteria_id'] = criteria_id
    df_result = df_result.reset_index().set_index(['on_online_id', 'on_criteria_id', 'on_date'])

    asset_on_online_criteria.save(online_id, criteria_id,  df_result)

    total_turnover = sr_turnover.sum()

    return total_turnover

