#coding=utf8


import getopt
import string
import json
import os
import sys
sys.path.append('shell')
import click
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
import RiskManagement

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath

import traceback, code

@click.group()  
@click.pass_context
def riskmgr(ctx):
    '''risk management group
    '''
    pass


@riskmgr.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2012-07-15', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.option('--label-asset/--no-label-asset', default=True)
@click.option('--reshape/--no-reshape', default=True)
@click.option('--markowitz/--no-markowitz', default=True)
@click.pass_context
def simple(ctx, datadir, startdate, enddate, label_asset, reshape, markowitz):
    '''run risk management using simple strategy
    '''
    Const.datadir = datadir

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")        

    df_inc = pd.read_csv(datapath('port_pct.csv'),  index_col=['date'], parse_dates=['date'])
    df_nav = (df_inc + 1).cumprod()

    df_pos = pd.read_csv(datapath('port_weight.csv'),  index_col=['date'], parse_dates=['date'])

    df_timing = pd.read_csv(datapath('hs_gftd.csv'),  index_col=['date'], parse_dates=['date'], usecols=['date', 'trade_types'])
    df_timing = df_timing.rename(columns={'trade_types':'sh000300'})

    print df_nav.head()
    print df_pos.head()
    print df_timing.head()

    risk_mgr = RiskManagement.RiskManagement()
    df_result = risk_mgr.perform(df_nav[['sh000300']], df_pos[['sh000300']], df_timing[['sh000300']])

    df_result.to_csv(datapath('riskmgr_result.csv'))
