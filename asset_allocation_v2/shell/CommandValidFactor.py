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
from ipdb import set_trace

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund, asset_stock_factor, asset_mz_markowitz_pos
from db.asset_stock_factor import *
from db.asset_stock import *
from stock_factor import *
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
from asset import WaveletAsset

from scipy.stats import rankdata
import xgboost as xgb
from xgboost import DMatrix

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def vf(ctx):
    '''valid stock factor
    '''
    pass


@vf.command()
@click.pass_context
def valid_factor_update(ctx):
    '''valid stock factor update
    '''

    for lookback in range(1, 100):
    # for lookback in [1] + list(range(10, 110, 10)):
        # df_valid, df_asset_inc = cal_valid_factor_t_sta(lookback)
        df_valid, df_asset_inc = cal_valid_factor_return(lookback)
        df_wr = valid_factor_rank(df_valid, df_asset_inc)
        print(lookback, df_wr.mean())

    # start_date = '2010-02-01'
    # end_date = datetime.now().strftime('%Y-%m-%d')
    # end_date = '2018-08-01'
    # factor_ids = ['MZ.FA00%d0'%i for i in range(1, 10)] + ['MZ.FA10%d0'%i for i in range(1, 10)]
    # blacklist = ['MZ.FA1050', 'MZ.FA1060', 'MZ.FA1070', 'MZ.FA1090']
    # factor_ids = np.setdiff1d(factor_ids, blacklist)
    # df_asset_nav = load_stock_factor_day_nav(factor_ids, start_date, end_date)

    # lookback = 1
    # df_asset_inc = df_asset_nav.pct_change(lookback).dropna()
    # df_valid = cal_valid_factor_wavelet(lookback)
    # df_wr = valid_factor_rank(df_valid, df_asset_inc)
    # print(lookback, df_wr.mean())


@vf.command()
@click.pass_context
def valid_factor_compare(ctx):
    '''valid stock factor allocate compare
    '''

    start_date = '2012-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    trade_dates = ATradeDate.week_trade_date(start_date, end_date)

    df_pos_1 = asset_mz_markowitz_pos.load('MZ.MF0010')
    df_pos_2 = asset_mz_markowitz_pos.load('MZ.MF0060')

    df_pos_1 = df_pos_1.reindex(trade_dates).fillna(method = 'pad')
    df_pos_2 = df_pos_2.reindex(trade_dates).fillna(method = 'pad')

    df_pos_same = np.sign(df_pos_1) * np.sign(df_pos_2)
    df_pos_same = df_pos_same.sum(1) / 5
    print('same factor ratio: ', df_pos_same.mean())


def cal_valid_factor(lookback):
    '''calculate valid stock factor
    '''
    start_date = '2010-02-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    # end_date = '2018-08-01'
    factor_ids = ['MZ.FA00%d0'%i for i in range(1, 10)] + ['MZ.FA10%d0'%i for i in range(1, 10)]
    blacklist = ['MZ.FA1050', 'MZ.FA1060', 'MZ.FA1070', 'MZ.FA1090']
    factor_ids = np.setdiff1d(factor_ids, blacklist)
    df_asset_nav = load_stock_factor_nav(factor_ids, start_date, end_date)
    df_asset_inc = df_asset_nav.pct_change(lookback).dropna()
    df_valid = df_asset_inc.apply(lambda x: x.nlargest(5), 1)
    df_valid = df_valid.fillna(0.0)
    df_valid[df_valid != 0.0] = 1.0

    return df_valid, df_asset_inc


def cal_valid_factor_xgboost(lookback):
    '''calculate valid stock factor
    '''
    forcast_period = 1
    train_num = 100
    start_date = '2008-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    factor_ids = ['MZ.FA00%d0'%i for i in range(1, 10)] + ['MZ.FA10%d0'%i for i in range(1, 10)]
    blacklist = ['MZ.FA1050', 'MZ.FA1060', 'MZ.FA1070', 'MZ.FA1090']
    factor_ids = np.setdiff1d(factor_ids, blacklist)
    df_asset_nav = load_stock_factor_nav(factor_ids, start_date, end_date)
    df_asset_inc = df_asset_nav.pct_change(lookback).dropna()
    df_asset_inc_day = df_asset_nav.pct_change(forcast_period).reindex(df_asset_inc.index)
    test_dates = df_asset_inc.index[df_asset_inc.index >= '2011-01-01']
    # test_dates = df_asset_inc.index[df_asset_inc.index >= '2018-05-01']
    df_valid = pd.DataFrame(columns = df_asset_inc.columns)
    for date, next_date in zip(test_dates[:-forcast_period], test_dates[forcast_period:]):
        # print(date)

        rank_pred = []
        for factor_id in factor_ids:
            train_x = df_asset_inc[df_asset_inc.index <= date].iloc[-train_num:]
            train_y = df_asset_inc_day[df_asset_inc_day.index <= next_date].iloc[forcast_period:].iloc[-train_num:]
            train_y = train_y.apply(rankdata, 1).loc[:, [factor_id]]
            train_dmatrix = DMatrix(train_x, train_y)
            test_x = df_asset_inc.loc[[next_date]]
            test_dmatrix = DMatrix(test_x)

            # params = {'objective': 'rank:pairwise', 'eta': 0.1, 'gamma': 1.0, 'min_child_weight': 1, 'max_depth': 6, 'silent': True}
            params = {'objective': 'rank:pairwise', 'silent': True}
            # xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4, evals=[(valid_dmatrix, 'validation')])
            xgb_model = xgb.train(params, train_dmatrix, num_boost_round=4)
            pred = xgb_model.predict(test_dmatrix)[0]
            rank_pred.append(pred)
        df_valid.loc[next_date] = rank_pred

    df_valid = df_valid.apply(lambda x: x.nlargest(5), 1)
    df_valid = df_valid.fillna(0.0)
    df_valid[df_valid != 0.0] = 1.0

    df_asset_nav = load_stock_factor_nav(factor_ids, start_date, end_date)
    df_asset_inc = df_asset_nav.pct_change()
    df_asset_inc = df_asset_inc.reindex(df_valid.index)

    return df_valid, df_asset_inc


def cal_valid_factor_ic(lookback):
    '''calculate valid stock factor
    '''
    start_date = '2010-02-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    # end_date = '2018-08-01'
    trade_dates = ATradeDate.week_trade_date(start_date, end_date)
    factor_ids_positive = ['MZ.FA00%d0'%i for i in range(1, 10)]
    factor_ids_negative = ['MZ.FA10%d0'%i for i in range(1, 10)]
    factor_ids = factor_ids_positive + factor_ids_negative
    blacklist = ['MZ.FA1050', 'MZ.FA1060', 'MZ.FA1070', 'MZ.FA1090']
    factor_ids_valid = np.setdiff1d(factor_ids, blacklist)

    df_factor_ic = asset_stock_factor.load_stock_factor_ic(begin_date = start_date, end_date = end_date)
    df_factor_ic = df_factor_ic.unstack().T
    df_factor_ic.index = df_factor_ic.index.levels[1]

    df_factor_ic.columns = factor_ids_positive
    df_factor_ic_negative = -df_factor_ic
    df_factor_ic_negative.columns = factor_ids_negative
    df_asset_ic = pd.concat([df_factor_ic, df_factor_ic_negative], 1)
    df_asset_ic = df_asset_ic.loc[:, factor_ids_valid]
    df_asset_ic = df_asset_ic.reindex(trade_dates)
    df_asset_ic = df_asset_ic.rolling(lookback).mean().abs().dropna()

    df_valid = df_asset_ic.apply(lambda x: x.nlargest(5), 1)
    df_valid = df_valid.fillna(0.0)
    df_valid[df_valid != 0.0] = 1.0

    df_asset_nav = load_stock_factor_nav(factor_ids, start_date, end_date)
    df_asset_inc = df_asset_nav.pct_change()
    df_asset_inc = df_asset_inc.reindex(df_valid.index)

    return df_valid, df_asset_inc


def cal_valid_factor_return(lookback):
    '''calculate valid stock factor
    '''
    start_date = '2010-02-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    # end_date = '2018-08-01'
    trade_dates = ATradeDate.week_trade_date(start_date, end_date)
    factor_ids_positive = ['MZ.FA00%d0'%i for i in range(1, 10)]
    factor_ids_negative = ['MZ.FA10%d0'%i for i in range(1, 10)]
    factor_ids = factor_ids_positive + factor_ids_negative
    blacklist = ['MZ.FA1050', 'MZ.FA1060', 'MZ.FA1070', 'MZ.FA1090']
    factor_ids_valid = np.setdiff1d(factor_ids, blacklist)

    df_factor_ic = asset_stock_factor.load_stock_factor_return(begin_date = start_date, end_date = end_date)
    df_factor_ic = df_factor_ic.unstack().T
    df_factor_ic.index = df_factor_ic.index.levels[1]

    df_factor_ic.columns = factor_ids_positive
    df_factor_ic_negative = -df_factor_ic
    df_factor_ic_negative.columns = factor_ids_negative
    df_asset_ic = pd.concat([df_factor_ic, df_factor_ic_negative], 1)
    df_asset_ic = df_asset_ic.loc[:, factor_ids_valid]
    df_asset_ic = df_asset_ic.reindex(trade_dates)
    df_asset_ic = df_asset_ic.rolling(lookback).mean().abs().dropna()

    df_valid = df_asset_ic.apply(lambda x: x.nlargest(5), 1)
    df_valid = df_valid.fillna(0.0)
    df_valid[df_valid != 0.0] = 1.0

    df_asset_nav = load_stock_factor_nav(factor_ids, start_date, end_date)
    df_asset_inc = df_asset_nav.pct_change()
    df_asset_inc = df_asset_inc.reindex(df_valid.index)

    return df_valid, df_asset_inc


def cal_valid_factor_t_sta(lookback):
    '''calculate valid stock factor
    '''
    start_date = '2010-02-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    # end_date = '2018-08-01'
    trade_dates = ATradeDate.week_trade_date(start_date, end_date)
    factor_ids_positive = ['MZ.FA00%d0'%i for i in range(1, 10)]
    factor_ids_negative = ['MZ.FA10%d0'%i for i in range(1, 10)]
    factor_ids = factor_ids_positive + factor_ids_negative
    blacklist = ['MZ.FA1050', 'MZ.FA1060', 'MZ.FA1070', 'MZ.FA1090']
    factor_ids_valid = np.setdiff1d(factor_ids, blacklist)

    df_factor_ic = asset_stock_factor.load_stock_factor_t_sta(begin_date = start_date, end_date = end_date)
    df_factor_ic = df_factor_ic.unstack().T
    df_factor_ic.index = df_factor_ic.index.levels[1]

    df_factor_ic.columns = factor_ids_positive
    df_factor_ic_negative = -df_factor_ic
    df_factor_ic_negative.columns = factor_ids_negative
    df_asset_ic = pd.concat([df_factor_ic, df_factor_ic_negative], 1)
    df_asset_ic = df_asset_ic.loc[:, factor_ids_valid]
    df_asset_ic = df_asset_ic.reindex(trade_dates)
    df_asset_ic = df_asset_ic.rolling(lookback).mean().abs().dropna()

    df_valid = df_asset_ic.apply(lambda x: x.nlargest(5), 1)
    df_valid = df_valid.fillna(0.0)
    df_valid[df_valid != 0.0] = 1.0

    df_asset_nav = load_stock_factor_nav(factor_ids, start_date, end_date)
    df_asset_inc = df_asset_nav.pct_change()
    df_asset_inc = df_asset_inc.reindex(df_valid.index)

    return df_valid, df_asset_inc


def cal_valid_factor_wavelet(lookback):
    '''calculate valid stock factor
    '''
    start_date = '2010-02-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    # end_date = '2018-08-01'
    factor_ids = ['MZ.FA00%d0'%i for i in range(1, 10)] + ['MZ.FA10%d0'%i for i in range(1, 10)]
    blacklist = ['MZ.FA1050', 'MZ.FA1060', 'MZ.FA1070', 'MZ.FA1090']
    factor_ids = np.setdiff1d(factor_ids, blacklist)
    df_asset_nav = load_stock_factor_wavelet_nav(factor_ids, start_date, end_date)
    df_asset_inc = df_asset_nav.pct_change(lookback).dropna()
    df_valid = df_asset_inc.apply(lambda x: x.nlargest(5), 1)
    df_valid = df_valid.fillna(0.0)
    df_valid[df_valid != 0.0] = 1.0

    return df_valid


def valid_factor_nav(df_valid, df_asset_inc):

    df_pos = df_valid.shift(1) / 5
    df_inc = (df_pos * df_asset_inc).sum(1)
    df_nav = (1 + df_inc).cumprod()

    return df_nav


def valid_factor_rank(df_valid, df_asset_inc):

    df_baseline = Asset.load_nav_series('120000084')
    df_baseline = df_baseline.reindex(df_valid.index).pct_change()
    df_valid = df_valid.shift(1)
    df_valid_ret = (df_valid * df_asset_inc).dropna(how = 'all')
    df_valid_ret = df_valid_ret.replace(0.0, np.nan)
    dates = df_valid_ret.index
    dates = dates[dates >=  '2012-01-01']
    win_ratios = []
    for date in dates:
        factor_ret = df_valid_ret.loc[date].dropna()
        csi800_ret = df_baseline.loc[date]
        win_num = len(factor_ret[factor_ret > csi800_ret])
        win_ratio = win_num / 5
        win_ratios.append(win_ratio)

    df_wr = pd.Series(data = win_ratios, index = dates)
    # print('Win Ratio: %.4f'%df_wr.mean())

    return df_wr


def load_stock_factor_nav(factor_ids, start_date, end_date):

    trade_dates = ATradeDate.week_trade_date(start_date, end_date)
    # trade_dates = ATradeDate.trade_date(start_date, end_date)
    asset_navs = {}
    for factor_id in factor_ids:
        asset_navs[factor_id] = Asset.load_nav_series(factor_id, reindex = trade_dates)

    df_asset_navs = pd.DataFrame(asset_navs)

    return df_asset_navs


def load_stock_factor_day_nav(factor_ids, start_date, end_date):

    trade_dates = ATradeDate.trade_date(start_date, end_date)
    asset_navs = {}
    for factor_id in factor_ids:
        asset_navs[factor_id] = Asset.load_nav_series(factor_id, reindex = trade_dates)

    df_asset_navs = pd.DataFrame(asset_navs)

    return df_asset_navs


def load_stock_factor_wavelet_nav(factor_ids, start_date, end_date):

    # trade_dates = ATradeDate.week_trade_date(start_date, end_date)
    # trade_dates = ATradeDate.trade_date(start_date, end_date)
    asset_navs = {}
    for factor_id in factor_ids:
        wavelet_asset = WaveletAsset(factor_id, 2)
        asset_navs[factor_id] = wavelet_asset.nav(begin_date = start_date, end_date = end_date)

    df_asset_navs = pd.DataFrame(asset_navs)

    return df_asset_navs




