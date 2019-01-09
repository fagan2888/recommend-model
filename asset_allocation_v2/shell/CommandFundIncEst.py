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
import copy
from ipdb import set_trace
from Const import datapath
from tabulate import tabulate
sys.path.append('shell')
from db import asset_fund_inc_estimate
from fund_inc_estimation import *


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

    fiesp = FundIncEstSkPos(begin_date='20171201')
    df_fiesp = fiesp.estimate_fund_inc()
    df_fiesp = df_fiesp.stack().to_frame(name='sk_pos')

    fieip = FundIncEstIxPos(begin_date='20171201')
    df_fieip = fieip.estimate_fund_inc()
    df_fieip = df_fieip.stack().to_frame(name='ix_pos')

    df_fie = df_fiesp.join(df_fieip, how='outer')
    df_fie['mix'] = df_fie.mean(axis='columns')

    asset_fund_inc_estimate.update_fund_inc_estimate(copy.deepcopy(df_fie))

    fiem = FundIncEstMix(begin_date='20180101')
    df_fiem = fiem.estimate_fund_inc()
    df_fiem = df_fiem.stack().to_frame(name='mix')

    df_fie = df_fie.join(df_fiem, how='left', lsuffix='_old', rsuffix='_new')
    df_fie['mix'] = df_fie.apply(lambda x: x.mix_old if x.isna().mix_new else x.mix_new, axis='columns')
    df_fie = df_fie.drop(['mix_old', 'mix_new'], axis='columns')
    df_fie = df_fie.loc[pd.Timestamp('20180101'):]

    asset_fund_inc_estimate.update_fund_inc_estimate(df_fie)


@fie.command()
@click.pass_context
def fund_inc_estimate_update(ctx):
    '''fund inc estimate update
    '''

    begin_date = asset_fund_inc_estimate.load_date_last_updated() + pd.Timedelta('1d')

    fiesp = FundIncEstSkPos(begin_date=begin_date)
    df_fiesp = fiesp.estimate_fund_inc()
    df_fiesp = df_fiesp.stack().to_frame(name='sk_pos')

    fieip = FundIncEstIxPos(begin_date=begin_date)
    df_fieip = fieip.estimate_fund_inc()
    df_fieip = df_fieip.stack().to_frame(name='ix_pos')

    df_fie = df_fiesp.join(df_fieip, how='outer')
    df_fie['mix'] = df_fie.mean(axis='columns')

    asset_fund_inc_estimate.update_fund_inc_estimate(copy.deepcopy(df_fie), begin_date)

    fiem = FundIncEstMix(begin_date=begin_date)
    df_fiem = fiem.estimate_fund_inc()
    df_fiem = df_fiem.stack().to_frame(name='mix')

    df_fie = df_fie.join(df_fiem, how='left', lsuffix='_old', rsuffix='_new')
    df_fie['mix'] = df_fie.apply(lambda x: x.mix_old if x.isna().mix_new else x.mix_new, axis='columns')
    df_fie = df_fie.drop(['mix_old', 'mix_new'], axis='columns')

    asset_fund_inc_estimate.update_fund_inc_estimate(df_fie, begin_date)

