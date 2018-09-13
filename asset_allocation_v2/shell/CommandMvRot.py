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
from scipy.stats import spearmanr
from ipdb import set_trace

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund, asset_fund_pos, asset_fund
from db.asset_stock_factor import *
from db.asset_stock import *
from stock_factor import *
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
from pyspark import SparkContext


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def mr(ctx):
    ''' Market Value Rotate
    '''


@mr.command()
@click.pass_context
def mv_pos_update(ctx):
    '''update fund market value position
    '''

    cal_mv_pos_days()
    # fund_select_rotate()


@mr.command()
@click.pass_context
def fund_search(ctx):
    '''industry position update
    '''

    # fund_select_rotate()
    df_mv_rot = pd.read_csv(
        'data/fund/df_mv_rot.csv',
        index_col=['date'],
        parse_dates=['date'],
    )
    df = df_mv_rot[df_mv_rot.index < '2017']
    df = df.tail(24).dropna(1).mean().sort_values()
    set_trace()

    # fund_info = base_ra_fund.load(codes=fund_list.index)
    # fund_info = fund_info.set_index('ra_code')
    # fund_info = fund_info[['ra_name']]
    # fund_list = pd.merge(fund_list, fund_info, left_index=True, right_index=True, how='inner')


def cal_mv_pos_days():
    '''cal industry position
    '''

    start_date = '2010-01-01'
    end_date = '2018-08-01'
    factor_ids = ['120000051', '120000002']
    df_nav_base = load_ind_nav(factor_ids, start_date, end_date)

    fund_nif = asset_fund.load_type_fund(l2codes=['200101']).index
    df_nav_fund = base_ra_fund_nav.load_daily(start_date, end_date, codes=fund_nif)

    df_inc_base = df_nav_base.pct_change().dropna(how='all')
    df_inc_fund = df_nav_fund.pct_change().dropna(how='all')

    lookback = 1
    dates = df_inc_base.resample('M').last().index
    dates = dates
    df_pos = None
    for date, pdate in zip(dates[:-lookback], dates[lookback:]):
        print(date)

        base_inc = df_inc_base.loc[date:pdate]
        fund_inc = df_inc_fund.loc[date:pdate]
        fund_inc = fund_inc.dropna(1)

        fund_codes = fund_inc.columns
        df_pos_day = pd.DataFrame(columns=factor_ids)
        for fund_code in fund_codes:
            fund_pos = cal_ind_pos(base_inc, fund_inc[fund_code])
            df_pos_day.loc[fund_code] = fund_pos

        df_pos_day = df_pos_day.stack().reset_index()
        df_pos_day.columns = ['fund_id', 'index_id', 'position']
        df_pos_day['trade_date'] = pdate
        df_pos_day = df_pos_day.set_index(['fund_id', 'index_id', 'trade_date'])
        if df_pos is None:
            df_pos = df_pos_day
        else:
            df_pos = pd.concat([df_pos, df_pos_day])

    # asset_fund_pos.update_fund_pos(df_pos)
    df_pos.to_csv('data/fund/fund_mv_pos.csv')


def cal_ind_pos(ind_inc, fund_inc):

    ind_num = ind_inc.shape[1]

    w0 = np.array([1.0 / ind_num]*ind_num)

    cons = (
        {'type': 'ineq', 'fun': lambda x: -sum(x) + 1.0},
        {'type': 'ineq', 'fun': lambda x: sum(x) - 0.8},
    )

    bnds = tuple([(0.0, 1.0) for i in range(ind_num)])

    res = minimize(fund_inc_objective, w0, args=[ind_inc, fund_inc], method='SLSQP', constraints=cons, options={'disp': False}, bounds=bnds)

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
        asset_navs[factor_id] = Asset.load_nav_series(factor_id, reindex=trade_dates)

    df_asset_nav = pd.DataFrame(asset_navs)

    return df_asset_nav


def fund_select_rotate():

    df_pos_cal = pd.read_csv(
        'data/fund/fund_mv_pos.csv',
        parse_dates=['trade_date'],
        dtype={'fund_id': str, 'index_id': str}
    )
    df_pos_cal = df_pos_cal.set_index(['fund_id', 'index_id', 'trade_date'])

    start_date = '2010-01-01'
    end_date = '2018-08-01'
    factor_ids = ['120000051', '120000002']
    df_nav_ind = load_ind_nav(factor_ids, start_date, end_date)
    df_ret_ind = df_nav_ind.pct_change().fillna(0.0)
    df_ret_ind = df_ret_ind.resample('m').sum()

    fund_ids = df_pos_cal.index.levels[0]
    df_rank = {}
    for fund_id in fund_ids:
        print(fund_id)
        fund_pos_cal = df_pos_cal.loc[fund_id]
        fund_pos_cal = fund_pos_cal.unstack()
        fund_pos_cal.columns = fund_pos_cal.columns.get_level_values(1)
        fund_pos_cal = fund_pos_cal.T
        fund_pos_cal = fund_pos_cal.shift(1).dropna()
        tmp_ret_ind = df_ret_ind.loc[fund_pos_cal.index, fund_pos_cal.columns]
        df_rank_fund = pd.Series()
        for date in fund_pos_cal.index:
            # tmp_rank_corr = spearmanr(tmp_ret_ind.loc[date], fund_pos_cal.loc[date])[0]
            tmp_rank_corr = (tmp_ret_ind.loc[date] * fund_pos_cal.loc[date]).sum()
            if np.isnan(tmp_rank_corr):
                tmp_rank_corr = 0
            df_rank_fund.loc[date] = tmp_rank_corr
        df_rank[fund_id] = df_rank_fund

    df_rank_corr = pd.DataFrame(df_rank)
    df_rank_corr.to_csv('data/fund/df_mv_rot.csv', index_label='date')

    return df_rank_corr

