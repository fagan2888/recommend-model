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
from db import asset_fund_inc_estimate
from fund_inc_estimate import *


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def fie(ctx):
    ''' fund inc estimate
    '''

    pass


# @fie.command()
# @click.pass_context
def fund_inc_estimate_create():
    '''fund inc estimate create
    '''

    fiesp = FundIncEstSkPos(begin_date='20181101')
    df_fiesp = fiesp.estimate_fund_inc()
    df_fiesp = df_fiesp.stack().to_frame(name='sk_pos')

    fieip = FundIncEstIxPos(begin_date='20181101')
    df_fieip = fieip.estimate_fund_inc()
    df_fieip = df_fieip.stack().to_frame(name='ix_pos')

    df_fie = df_fiesp.join(df_fieip, how='outer')

    asset_fund_inc_estimate.update_fund_inc_estimate(df_fie)

    fiem = FundIncEstMix('20181201')
    df_fiem = fiem.estimate_fund_inc()
    df_fiem = df_fiem.stack().to_frame(name='mix')

    df_fie = df_fie.join(df_fiem, how='left')

    asset_fund_inc_estimate.update_fund_inc_estimate(df_fie, begin_date='20181201')

if 1==1:
    fund_inc_estimate_create()

# 再写一个update_all_methods

set_trace()

@fie.command()
@click.pass_context
def fund_inc_estimate_update(ctx):
    '''fund inc estimate update
    '''

    df_fund_inc_estimate = fund_inc_estimate()
    # df_fund_inc_estimate.index.names = []
    # df_fund_inc_estimate.columns = []
    asset_fund_inc_estimate.update_fund_inc_estimate(df_fund_inc_estimate)


