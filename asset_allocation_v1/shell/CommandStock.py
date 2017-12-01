#coding=utf8


import string
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import os
import time
import logging
import re
import Const
import DFUtil
import DBData
import util_numpy as npu
import Portfolio as PF
from TimingWavelet import TimingWt
import multiprocessing
from multiprocessing import Manager

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from tabulate import tabulate
from db import database, asset_mz_markowitz, asset_mz_markowitz_alloc, asset_mz_markowitz_argv,  asset_mz_markowitz_asset, asset_mz_markowitz_criteria, asset_mz_markowitz_nav, asset_mz_markowitz_pos, asset_mz_markowitz_sharpe, asset_wt_filter_nav
from db import asset_ra_pool, asset_ra_pool_nav, asset_rs_reshape, asset_rs_reshape_nav, asset_rs_reshape_pos
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates, base_exchange_rate_index_nav, base_ra_stock, base_ra_stock_nav
from util import xdict
from util.xdebug import dd
from mxnet import nd

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
def stock():

    '''stock group
    '''
    pass

@stock.command()
@click.option('--output', 'optoutput', default=None, help=u'output path')
@click.pass_context
def export_navaf(ctx, optoutput):
    '''
    export stock nav
    '''
    df = base_ra_stock_nav.closeaf()
    df = df.reset_index()
    df = df.set_index(['date', 'globalid'])
    df = df.unstack()
    df.columns = df.columns.get_level_values(1)
    print df.tail()
    if optoutput is not None:
        df.to_csv(optoutput.strip())


@stock.command()
@click.option('--input', 'optinput', default=None, help=u'stock nav path')
@click.pass_context
def navaf_rnn(ctx, optinput):

    nav_df = pd.read_csv(optinput.strip(), index_col = ['date'], parse_dates = ['date'])
    nav_df = nav_df.replace(0, np.nan)
    nav_df = nav_df.fillna(method = 'pad')
    inc_df = nav_df.pct_change().fillna(0.0)
    print inc_df
    inc_df.to_csv('inc.csv')
