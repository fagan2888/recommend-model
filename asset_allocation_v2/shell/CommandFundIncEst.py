#coding=utf-8
'''
Created at Jan. 3, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import getopt
import string
import json
import os
import sys
import logging
import click
import config
import pandas as pd
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import spearmanr
from ipdb import set_trace
from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from tabulate import tabulate
sys.path.append('shell')
from db import database, asset_fund_inc_estimate
from fund_inc_estimate import *


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def fie(ctx):
    ''' fund inc estimate
    '''

    pass


@fie.command()
@click.pass_context
def fund_inc_estimate_create(ctx):
    '''fund inc estimate create
    '''

    begin_date = pd.Timestamp('20170101')
    end_date = pd.Timeestamp(pd.Timestamp.now().strftime('%Y%m%d'))
    dates = pd.Index([begin_date, end_date])

    df_new = pd.DataFrame()
    for date, next_date in zip(dates[1:], dates[:-1]):

        next_date -= pd.Timedelta('1d')

        a = FundIncEstSkPos(begin_date=date, end_date=next_date)
        aa = a.estimate_fund_inc()
        aa = aa.stack().to_frame()

        b = FundIncIxPos(begin_date=date, end_date=next_date)
        bb = b.estimate_fund_inc()
        bb = bb.stack().to_frame()

        cc = aa.join(bb, how='outer')

        df_new = pd.concat([df_new, df])

    asset_fund_inc_estimate.update_fund_inc_estimate(df_new)


if 1==1:
    fund_inc_estimate_create()
# 写一个create
# 逻辑
# 先更新sk_pos indx_pos 空的mix
# 然后更新

# 再写一个update_all_methods

@fie.command()
@click.pass_context
def fund_inc_estimate_update(ctx):
    '''fund inc estimate update
    '''

    df_fund_inc_estimate = fund_inc_estimate()
    # df_fund_inc_estimate.index.names = []
    # df_fund_inc_estimate.columns = []
    asset_fund_inc_estimate.update_fund_inc_estimate(df_fund_inc_estimate)


