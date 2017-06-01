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
from db import *
from util.xdebug import dd

import traceback, code

logger = logging.getLogger(__name__)

@click.group()  
@click.pass_context
def export(ctx):
    ''' generate portfolios
    '''
    pass;
    
@export.command()
@click.option('--inst', 'optinst', help=u'portfolio to exprot (e.g. 2016120700:10,20161207:5)')
@click.option('--index', 'optindex', help=u'index to export (e.g. 120000001,120000002)')
@click.option('--composite', 'optcomposite', help=u'composite asset to export (e.g. 20001,2002)')
@click.option('--fund', 'optfund', help=u'fund to export (e.g. 20001,2002)')
@click.option('--pool', 'optpool', help=u'fund pool to export (e.g. 921001:0,92101:11)')
@click.option('--online', 'optonline', help=u'online model')
@click.option('--highlow', 'opthighlow', help=u'highlow to export(e.g. 70052400,70052401)')
@click.option('--portfolio', 'optportfolio', help=u'portfolio to export(e.g. 80052400:9)')
@click.option('--timing', 'opttiming', help=u'timing to export(e.g. 21400501:8)')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.option('--start-date', 'optstartdate', default='2012-07-27', help=u'start date to calc')
@click.option('--end-date', 'optenddate', help=u'end date to calc')
# @click.option('--tools', '-t', type=click.Choice(['tool1', 'tool2', 'tool3']), multiple=True)
@click.option('--datetype', 'optdatetype', type=click.Choice(['t', 'n']), default='t', help=u'date type(t: trade date; n: nature date)')
@click.option('--output', '-o', type=click.Path(), help=u'output file')
@click.pass_context
def nav(ctx, optinst, optindex, optcomposite, optfund, optpool, optstartdate, optenddate, optlist, optdatetype, optonline, optportfolio, opttiming, opthighlow, output):
    '''run constant risk model
    '''    
    if not optenddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        optenddate = yesterday.strftime("%Y-%m-%d")

    if optdatetype == 't':
        dates = base_trade_dates.load_index(optstartdate, optenddate)
    else:
        print "not implement!!"
        return 0;

    data = {}
    if optinst is not None:
        insts = [s.strip() for s in optinst.split(',')]
        for inst in insts:
            (inst_id, alloc_id, xtype) = [s.strip() for s in inst.split(':')]
            data[inst] = database.asset_allocation_instance_nav_load_series(
                inst_id, alloc_id, xtype, reindex=dates, begin_date=optstartdate, end_date=optenddate)

    if optindex is not None:
        indexs = [s.strip() for s in optindex.split(',')]
        for e in indexs:
            data[e] = base_ra_index_nav.load_series(
                e, reindex=dates, begin_date=optstartdate, end_date=optenddate)

    if optcomposite is not None:
        composites = [s.strip() for s in optcomposite.split(',')]
        for e in composites:
            data[e] = database.asset_ra_composite_asset_load_series(
                e, reindex=dates, begin_date=optstartdate, end_date=optenddate)

    if optfund is not None:
        funds = [s.strip() for s in optfund.split(',')]
        for e in funds:
            data[e] = base_ra_fund_nav.load_series(
                e, reindex=dates, begin_date=optstartdate, end_date=optenddate)
        
    if optpool is not None:
        pools = [s.strip() for s in optpool.split(',')]
        for e in pools:
            (pool_id, category, xtype) = [s.strip() for s in e.split(':')]
            data[e] = database.asset_ra_pool_nav_load_series(
                pool_id, category, xtype, reindex=dates, begin_date=optstartdate, end_date=optenddate)
    if optonline is not None:
        allocs = [s.strip() for s in optonline.split(',')]
        for e in allocs:
            (alloc, xtype) = [s.strip() for s in e.split(':')]
            data["online:%s" % (e)] = database.asset_risk_asset_allocation_nav_load_series(
                alloc, xtype, reindex=dates, begin_date=optstartdate, end_date=optenddate)

    if optportfolio is not None:
        allocs = [s.strip() for s in optportfolio.split(',')]
        for e in allocs:
            (alloc, xtype) = [s.strip() for s in e.split(':')]
            data[e] = asset_ra_portfolio_nav.load_series(
                alloc, xtype, reindex=dates, begin_date=optstartdate, end_date=optenddate)

    if opttiming is not None:
        allocs = [s.strip() for s in opttiming.split(',')]
        for e in allocs:
            data[e] = asset_tc_timing_nav.load_series(
                e, reindex=dates, begin_date=optstartdate, end_date=optenddate)

    if opthighlow is not None:
        allocs = [s.strip() for s in opthighlow.split(',')]
        for e in allocs:
            data[e] = asset_mz_highlow_nav.load_series(
                e, reindex=dates, begin_date=optstartdate, end_date=optenddate)

    df_result = pd.concat(data, axis=1)

    if output is not None:
        path = output
    else:
        path = datapath('export-nav.csv')
        
    df_result.to_csv(path)

    print "export nav to file %s" % (path)

