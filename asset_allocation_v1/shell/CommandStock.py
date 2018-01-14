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
from db.asset_stock import *
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker

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
    #print df.tail()
    if optoutput is not None:
        df.to_csv(optoutput.strip())


@stock.command()
@click.option('--input', 'optinput', default=None, help=u'stock nav path')
@click.pass_context
def navaf_rnn_data(ctx, optinput):

    nav_df = pd.read_csv(optinput.strip(), index_col = ['date'], parse_dates = ['date'])
    nav_df = nav_df.replace(0, np.nan)
    nav_df = nav_df.interpolate()
    inc_df = nav_df.pct_change()

    for i in range(0 ,len(inc_df)):
        mean = inc_df.iloc[i].mean()
        std = inc_df.iloc[i].std()
        inc_df.iloc[i] = (inc_df.iloc[i] - mean) / std

    #print inc_df
    X = []
    Y = []
    interval = 60
    for code in inc_df.columns:
        ser = inc_df[code]
        for i in range(interval, len(ser) - 1, interval):
            x = ser.iloc[i - interval : i]
            y = ser.iloc[i - interval + 1 : i + 1]
            if len(x) <= len(x.dropna()) and len(y) <= len(y.dropna()):
                X.append(x.ravel())
                Y.append(y.ravel())

        print code, 'Done'

    with open('stock_x.csv', 'w') as f:
        for x in X:
            f.writelines(','.join([str(n) for n in x]) + '\n')
        f.close()

    with open('stock_y.csv', 'w') as f:
        for y in Y:
            f.writelines(','.join([str(n) for n in y]) + '\n')
        f.close()

@stock.command()
@click.option('--x', 'optx', default=None, help=u'stock x path')
@click.option('--y', 'opty', default=None, help=u'stock y path')
@click.pass_context
def rnn(ctx, optx, opty):

    with open('stock_x.csv', 'r') as f:
        X = f.readlines()
        print len(X)

    with open('stock_y.csv', 'r') as f:
        Y = f.readlines()
        print len(Y)

#所有股票代码
def all_stock_info():

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    all_stocks = pd.read_sql(session.query(ra_stock.globalid, ra_stock.sk_secode, ra_stock.sk_compcode, ra_stock.sk_name).statement, session.bind, index_col = ['sk_secode'])
    session.commit()
    session.close()

    return all_stocks


@stock.command()
@click.pass_context
def stock_wavelet(ctx):

    all_stocks = all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    for i in range(0, len(all_stocks.index)):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        sql = session.query(tq_sk_dquoteindic.tradedate, tq_sk_dquoteindic.tcloseaf).filter(tq_sk_dquoteindic.secode == secode).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
        sr = quotation.tcloseaf
        wt = TimingWt(sr)
        filtered_data = wt.wavefilter(sr, 2)
        filtered_data = filtered_data.fillna(0.0)
        print filtered_data



