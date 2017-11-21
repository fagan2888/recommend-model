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
import util_numpy as npu
import TradeNav

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from tabulate import tabulate
from db import *
from isim import *
from util.xdebug import dd

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)  
@click.option('--full/--no-full', 'optfull', default=False, help=u'include all instance')
@click.option('--id', 'optid', help=u'specify investor id')
@click.option('--type', 'opttype', default='9', help=u'type type(8:with fee; 9:without fee')
@click.option('--end-date', 'optenddate', default=None, help=u'calc end date for nav')
@click.pass_context
def investor(ctx, optfull, optid, opttype, optenddate):

    '''generate final portolio
    '''
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        if optfull is False:
            ctx.invoke(nav, optid=optid, optenddate=optenddate)
            ctx.invoke(turnover, optid=optid)
        else:
            ctx.invoke(nav, optid=optid, optenddate=optenddate)
            ctx.invoke(turnover, optid=optid)
    else:
        # click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        pass


@investor.command()
@click.option('--id', 'optid', help=u'ids of investor to update')
@click.option('--type', 'opttype', default='9', help=u'type type(8:with fee; 9:without fee')
@click.option('--fee/--no-fee', 'optfee', default=True, help=u'specify with/without fee for type 8')
@click.option('--t0/--no-t0', 'optt0', default=False, help=u'specify use t+0 or not for type 8')
@click.option('--debug/--no-debug', 'optdebug', default=False, help=u'debug mode')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.option('--end-date', 'optenddate', default=None, help=u'calc end date for nav')
@click.pass_context
def nav(ctx, optid, optlist, opttype, optdebug, optfee, optt0, optenddate):
    ''' calc pool nav and inc
    '''
    if optid is not None:
        investors = [s.strip() for s in optid.split(',')]
    else:
        if 'investor' in ctx.obj:
            investors = [str(ctx.obj['investor'])]
        else:
            investors = None

    types = [int(s.strip()) for s in opttype.split(',')]

    if optenddate is not None:
        enddate = pd.to_datetime(optenddate)
    else:
        enddate = None
        
    df_investor = asset_is_investor.load(investors)

    if optlist:
        df_investor['is_name'] = df_investor['is_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_investor, headers='keys', tablefmt='psql')
        return 0

    for xtype in types:
        with click.progressbar(
            df_investor.iterrows(), length=len(df_investor.index),
                label='update nav'.ljust(30),
                item_show_func=lambda x: str(x[1]['globalid']) if x else None) as bar:
            for _, investor in bar:
                nav_update(investor, xtype, optdebug, optfee, optt0, enddate)
     
# def nav_update(alloc, fee, debug):
def nav_update(alloc, xtype, debug, optfee, optt0, enddate):
    alloc_id = alloc['globalid']
    # 加载仓位信息
    df_pos = asset_is_investor_pos.load_fund_pos(alloc_id)
    if df_pos.empty:
        click.echo(click.style("\nswarning: empty df_pos for alloc %s, skiped!" % (alloc_id), fg='yellow'))
        return
    
    df_pos.index.names=['ra_date', 'ra_fund_id']
    df_pos = df_pos.rename(columns={'is_fund_ratio': 'ra_fund_ratio'})

    if enddate is not None:
        max_date = enddate
    else:
        max_date = (datetime.now() - timedelta(days=1)) # yesterday

    # 计算复合资产净值
    if xtype == 8:
        xtype = 8
        df_pos = df_pos.loc[df_pos.index.get_level_values(0) >= '2012-07-27']
        tn = TradeNav.TradeNav(debug=debug, optfee=optfee, optt0=optt0)
        # tn.calc(df_pos, 100000)
        tn.calc(df_pos, 1)
        sr_nav_investor = pd.Series(tn.nav)
        sr_contrib = pd.concat(tn.contrib)
    else:
        xtype = 9
        sr_nav_investor = DFUtil.portfolio_nav2(df_pos, end_date=max_date)
        sr_contrib = pd.Series()

    df_result = sr_nav_investor.to_frame('is_nav')
    df_result.index.name = 'is_date'
    df_result['is_type'] = xtype
    df_result['is_inc'] = df_result['is_nav'].pct_change().fillna(0.0)
    df_result['is_investor_id'] = alloc_id
    df_result = df_result.reset_index().set_index(['is_investor_id', 'is_type', 'is_date'])

    asset_is_investor_nav.save(alloc_id, xtype, df_result)

    if not sr_contrib.empty:
        df_contrib = sr_contrib.to_frame('is_return_value')
        df_contrib.index.names=[u'is_date', u'is_return_type', u'is_fund_id']
        df_contrib['is_type'] = xtype
        df_contrib['is_investor_id'] = alloc_id
        df_contrib = df_contrib.reset_index().set_index(['is_investor_id', 'is_type', 'is_date', 'is_fund_id', 'is_return_type'])

        asset_is_investor_contrib.save(alloc_id, xtype, df_contrib)

@investor.command()
@click.option('--id', 'optid', help=u'ids of portfolio to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.pass_context
def turnover(ctx, optid, optlist):
    ''' calc investor turnover
    '''
    if optid is not None:
        investors = [s.strip() for s in optid.split(',')]
    else:
        if 'investor' in ctx.obj:
            investors = [str(ctx.obj['investor'])]
        else:
            investors = None

    df_investor = asset_is_investor.load(investors)

    if optlist:

        df_investor['is_name'] = df_investor['is_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_investor, headers='keys', tablefmt='psql')
        return 0
    
    data = []
    with click.progressbar(
            df_investor.iterrows(), length=len(df_investor.index),
            label='calc turnover'.ljust(30),
            item_show_func=lambda x:  str(x[1]['globalid']) if x else None) as bar:
        for _, investor in bar:
            turnover_update(investor)

def turnover_update(investor):
    investor_id = investor['globalid']
    # 加载仓位信息
    df = asset_is_investor_pos.load_fund_pos(investor_id)
    df = df.unstack()


    # 计算宽口换手率
    sr_turnover = DFUtil.calc_turnover(df)

    criteria_id = 6
    df_result = sr_turnover.to_frame('is_value')
    df_result['is_investor_id'] = investor_id
    df_result['is_criteria_id'] = criteria_id
    df_result = df_result.reset_index().set_index(['is_investor_id', 'is_criteria_id', 'is_date'])

    asset_is_investor_criteria.save(investor_id, criteria_id,  df_result)

    total_turnover = sr_turnover.sum()

    return total_turnover


@investor.command()
@click.option('--id', 'optid', help=u'ids of investor to update')
@click.option('--type', 'opttype', default='9', help=u'type type(8:with fee; 9:without fee')
@click.option('--fee/--no-fee', 'optfee', default=True, help=u'specify with/without fee for type 8')
@click.option('--t0/--no-t0', 'optt0', default=False, help=u'specify use t+0 or not for type 8')
@click.option('--debug/--no-debug', 'optdebug', default=False, help=u'debug mode')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.option('--end-date', 'optenddate', default=None, help=u'calc end date for nav')
@click.pass_context
def emulate(ctx, optid, optlist, opttype, optdebug, optfee, optt0, optenddate):
    ''' calc pool nav and inc
    '''
    if optid is not None:
        investors = [s.strip() for s in optid.split(',')]
    else:
        if 'investor' in ctx.obj:
            investors = [str(ctx.obj['investor'])]
        else:
            investors = None

    # types = [int(s.strip()) for s in opttype.split(',')]

    if optenddate is not None:
        enddate = pd.to_datetime(optenddate)
    else:
        enddate = None
        
    # df_investor = asset_is_investor.load(investors)

    # if optlist:
    #     df_investor['is_name'] = df_investor['is_name'].map(lambda e: e.decode('utf-8'))
    #     print tabulate(df_investor, headers='keys', tablefmt='psql')
    #     return 0

    # for xtype in types:
    #     with click.progressbar(
    #         df_investor.iterrows(), length=len(df_investor.index),
    #             label='update nav'.ljust(30),
    #             item_show_func=lambda x: str(x[1]['globalid']) if x else None) as bar:
    #         for _, investor in bar:
    #             nav_update(investor, xtype, optdebug, optfee, optt0, enddate)
    for investor in investors:
        emulate_update(investor, optdebug, optfee, optt0, enddate)
     
# def nav_update(alloc, fee, debug):
def emulate_update(uid, debug, optfee, optt0, enddate):

    # 加载仓位信息
    df_ts_order = trade_ts_order.load(uid, [3, 4, 6])

    if df_ts_order.empty:
        click.echo(click.style("\nswarning: empty df_ts_order for user: %s, skiped!" % (uid), fg='yellow'))
        return

    investor = Investor.Investor(df_ts_order)

    df_ts_order_fund = trade_ts_order_fund.load(uid)
    if df_ts_order_fund.empty:
        click.echo(click.style("\nswarning: empty df_ts_order_fund for user: %s, skiped!" % (uid), fg='yellow'))
        return
    
    policy = Policy.Policy(df_ts_order_fund)    

    emulator = InvestorShare.InvestorShare(investor, policy)

    df = emulator.run()
    
    dd(df)


