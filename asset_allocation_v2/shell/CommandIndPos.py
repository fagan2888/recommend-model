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
import time
from scipy.optimize import minimize
from ipdb import set_trace

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund, asset_fund_pos
from db.asset_stock_factor import *
from db.asset_stock import *
from stock_factor import *
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
from pyspark import SparkContext


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def ip(ctx):
    ''' industry position
    '''
    pass


@ip.command()
@click.pass_context
def industry_position_update(ctx):
    '''industry position update
    '''

    cal_ind_pos_days()


def cal_ind_pos_days():
    '''cal industry position
    '''

    start_date = '2010-01-01'
    end_date = '2018-08-01'
    factor_ids = ['1200000%2d'%i for i in range(52, 80)]
    df_nav_ind = load_ind_nav(factor_ids, start_date, end_date)

    # pool_codes = list(base_ra_fund.find_type_fund(1).ra_code.ravel())
    # df_nav_fund = base_ra_fund_nav.load_daily(start_date, end_date, codes = pool_codes)
    df_nav_fund = pd.read_csv('data/fund/df_nav_fund.csv', index_col = ['date'], parse_dates = ['date'])

    df_inc_ind = df_nav_ind.pct_change().dropna(how = 'all')
    df_inc_fund = df_nav_fund.pct_change().dropna(how = 'all')

    lookback = 3
    dates = df_inc_ind.resample('M').last().index
    dates = dates
    df_pos = None
    for date, pdate in zip(dates[:-lookback], dates[lookback:]):
        print(date)

        ind_inc = df_inc_ind.loc[date:pdate]
        fund_inc = df_inc_fund.loc[date:pdate]
        fund_inc = fund_inc.dropna(1)

        fund_codes = fund_inc.columns
        df_pos_day = pd.DataFrame(columns = factor_ids)
        for fund_code in fund_codes:
            fund_pos = cal_ind_pos(ind_inc, fund_inc[fund_code])
            df_pos_day.loc[fund_code] = fund_pos

        df_pos_day = df_pos_day.stack().reset_index()
        df_pos_day.columns = ['fund_id', 'index_id', 'position']
        df_pos_day['trade_date'] = pdate
        df_pos_day = df_pos_day.set_index(['fund_id', 'index_id', 'trade_date'])
        if df_pos is None:
            df_pos = df_pos_day
        else:
            df_pos = pd.concat([df_pos, df_pos_day])

    asset_fund_pos.update_fund_pos(df_pos)
    set_trace()


def cal_ind_pos(ind_inc, fund_inc):

    ind_num = ind_inc.shape[1]

    w0 = np.array([1.0 / ind_num]*ind_num)

    cons = (
        {'type': 'ineq', 'fun': lambda x: -sum(x) + 1.0},
        {'type': 'ineq', 'fun': lambda x: sum(x)},
    )

    bnds = tuple([(0.0, 1.0) for i in range(ind_num)])

    res = minimize(fund_inc_objective, w0, args=[ind_inc, fund_inc], method='SLSQP', constraints=cons, options={'disp': False}, bounds = bnds)

    return res.x


def fund_inc_objective(x, pars):

    ind_inc, fund_inc = pars
    fund_ret = fund_inc.values
    pre_ret = np.dot(ind_inc, x)
    loss = np.sum(np.power((fund_ret - pre_ret), 2))

    return loss


def load_ind_nav(factor_ids, start_date, end_date):

    trade_dates = ATradeDate.trade_date(start_date, end_date)
    asset_navs = {}
    for factor_id in factor_ids:
        asset_navs[factor_id] = Asset.load_nav_series(factor_id, reindex = trade_dates)

    df_asset_nav = pd.DataFrame(asset_navs)

    return df_asset_nav











