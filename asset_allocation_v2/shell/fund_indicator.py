# -*- coding: utf-8 -*-

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
from functools import partial
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from utils import get_today
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import distinct
from tabulate import tabulate
import statsmodels.api as sm
import statsmodels
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund, asset_fund, asset_mz_markowitz_pos
from factor import Factor
from asset import StockAsset, StockFundAsset
from db import asset_trade_dates
from db.asset_stock_factor import *
from db import asset_fund_factor
import math
import scipy.stats as stats
import json
from asset import Asset,FundAsset,StockFundAsset
from stock_factor import StockFactor
from functools import reduce
from trade_date import ATradeDate
from ipdb import set_trace
from multiprocessing import Pool

logger = logging.getLogger(__name__)


class FundIndicator(object):

    def __init__(self, start_date, end_date):

        self.start_date = start_date
        self.end_date = end_date
        self.lookbacks = [1, 3, 6, 12, 18, 24, 30]
        self.algos = {
            'return': self.cal_return_indicator,
            'std': self.cal_std_indicator,
            'downstd': self.cal_downstd_indicator,
            'beta': self.cal_beta_indicator,
            'sharpe': self.cal_sharpe_indicator,
            'sortino': self.cal_sortino_indicator,
            'jensen': self.cal_jensen_indicator,
        }

        # self.lookbacks = [12]
        # self.algos = {
            # 'std': self.cal_std_indicator,
        # }

        self.indicator_pool = {}

    def load_fund_nav(self, start_date, end_date):

        valid_funds = asset_fund.load_type_fund(l1codes=['2001']).index
        df_nav_fund = base_ra_fund_nav.load_daily(start_date, end_date, codes=valid_funds, fillna_limit=10)
        # df_nav_fund = df_nav_fund.resample('m').last()

        return df_nav_fund

    def load_base_nav(self, start_date, end_date):

        df = Asset.load_nav_series('120000016', begin_date=start_date, end_date=end_date)
        # df = df.resample('m').last()
        self.base_nav = df

        return df

    def cal_return_indicator(self, fund_nav, lookback, name):

        fund_nav = fund_nav.resample('m').last()
        fund_return = fund_nav.pct_change(lookback, limit=1)
        fund_return = fund_return.sub(fund_return.mean(1), 0).div(fund_return.std(1), 0)
        fund_return = fund_return.stack()
        fund_return.columns = [name]

        return fund_return

    def cal_std_indicator(self, fund_nav, lookback, name):

        fund_ret = fund_nav.pct_change(limit=1)
        fund_std = fund_ret.rolling(lookback*21).std()
        fund_std = fund_std.resample('m').last()
        fund_std = fund_std.sub(fund_std.mean(1), 0).div(fund_std.std(1), 0)
        fund_std = fund_std.stack()
        fund_std.columns = [name]

        return -fund_std

    def cal_downstd_indicator(self, fund_nav, lookback, name):

        fund_ret = fund_nav.pct_change(limit=1)
        fund_downstd = fund_ret.rolling(lookback*21).apply(downstd, raw=True)
        fund_downstd = fund_downstd.resample('m').last()
        fund_downstd = fund_downstd.sub(fund_downstd.mean(1), 0).div(fund_downstd.std(1), 0)
        fund_downstd = fund_downstd.stack()
        fund_downstd.columns = [name]

        return -fund_downstd

    def cal_beta_indicator(self, fund_nav, lookback, name):

        fund_ret = fund_nav.pct_change(limit=1)
        fund_ret_month = fund_ret.resample('m').last()
        dates = fund_ret_month.index
        fund_beta = pd.DataFrame()
        data = {}
        for fund in fund_nav.columns:
            data[fund] = {}
            for date, ndate in zip(dates[:-lookback], dates[lookback:]):
                y = fund_nav.loc[date:ndate, fund]
                x = self.base_nav.loc[date:ndate]
                if any(y.isna()):
                    data[fund][ndate] = np.nan
                else:
                    y = y.pct_change(limit=1).dropna()
                    x = x.pct_change(limit=1).dropna()
                    x = x.reindex(y.index)
                    x = sm.add_constant(x, 1)
                    mod = sm.OLS(y, x).fit()
                    data[fund][ndate] = mod.params.nav
        fund_beta = fund_beta.from_dict(data)
        fund_beta = fund_beta.sub(fund_beta.mean(1), 0).div(fund_beta.std(1), 0)
        fund_beta = fund_beta.stack()
        fund_beta.columns = [name]

        return -fund_beta

    def cal_sharpe_indicator(self, fund_nav, lookback, name):

        fund_return = fund_nav.pct_change(lookback, limit=1)
        fund_return = fund_return.resample('m').last()

        fund_ret = fund_nav.pct_change(limit=1)
        fund_std = fund_ret.rolling(lookback*21).std() * np.sqrt(21)
        fund_std = fund_std.resample('m').last()

        fund_sharpe = fund_return / fund_std
        fund_sharpe = fund_sharpe.sub(fund_sharpe.mean(1), 0).div(fund_sharpe.std(1), 0)
        fund_sharpe = fund_sharpe.stack()
        fund_sharpe.columns = [name]

        return fund_sharpe

    def cal_sortino_indicator(self, fund_nav, lookback, name):

        fund_return = fund_nav.pct_change(lookback, limit=1)
        fund_return = fund_return.resample('m').last()

        fund_ret = fund_nav.pct_change(limit=1)
        fund_downstd = fund_ret.rolling(lookback*21).apply(downstd, raw=True) * np.sqrt(21)
        fund_downstd = fund_downstd.resample('m').last()

        fund_sortino = fund_return / fund_downstd
        fund_sortino = fund_sortino.sub(fund_sortino.mean(1), 0).div(fund_sortino.std(1), 0)
        fund_sortino = fund_sortino.stack()
        fund_sortino.columns = [name]

        return fund_sortino

    def cal_jensen_indicator(self, fund_nav, lookback, name):

        fund_ret = fund_nav.pct_change(limit=1)
        fund_ret_month = fund_ret.resample('m').last()
        dates = fund_ret_month.index
        fund_jensen = pd.DataFrame()
        data = {}
        for fund in fund_nav.columns:
            data[fund] = {}
            for date, ndate in zip(dates[:-lookback], dates[lookback:]):
                y = fund_nav.loc[date:ndate, fund]
                x = self.base_nav.loc[date:ndate]
                if any(y.isna()):
                    data[fund][ndate] = np.nan
                else:
                    y = y.pct_change(limit=1).dropna()
                    x = x.pct_change(limit=1).dropna()
                    x = x.reindex(y.index)
                    x = sm.add_constant(x, 1)
                    mod = sm.OLS(y, x).fit()
                    data[fund][ndate] = mod.params.const
        fund_jensen = fund_jensen.from_dict(data)
        fund_jensen = fund_jensen.sub(fund_jensen.mean(1), 0).div(fund_jensen.std(1), 0)
        fund_jensen = fund_jensen.stack()
        fund_jensen.columns = [name]

        return fund_jensen

    def cal_indicator(self):

        self.load_base_nav(self.start_date, self.end_date)
        fund_nav = self.load_fund_nav(self.start_date, self.end_date)
        for algo_name, algo in self.algos.items():
            for lookback in self.lookbacks:
                name = '%s_%d' % (algo_name, lookback)
                tmp_indicator = algo(fund_nav, lookback, name)
                self.indicator_pool[name] = tmp_indicator
        self.indicator_df = pd.DataFrame(self.indicator_pool)
        self.indicator_df.to_csv('data/fund_indicator/fund_indicator.csv', index_label=['date', 'fund_id'])

    def cal_indicator_index_pos(self, indicator, fund_num):

        indicator_df = pd.read_csv('data/fund_indicator/fund_indicator.csv', dtype={'fund_id': object}, index_col=['date'], parse_dates=['date'])
        indicator_df = indicator_df.set_index('fund_id', append=True)
        # indicators = indicator_df.columns

        tmp_indicator = indicator_df[[indicator]]
        tmp_indicator = tmp_indicator.unstack()
        tmp_indicator.columns = tmp_indicator.columns.get_level_values(1)
        tmp_indicator = tmp_indicator.dropna(how='all')

        dates = tmp_indicator.index
        fund_ids = indicator_df.index.levels[1]
        df_pos = pd.DataFrame(data=np.zeros((len(dates), len(fund_ids))), columns=fund_ids, index=dates)

        for date in tmp_indicator.index:
            tmp_funds = tmp_indicator.loc[date]
            tmp_funds = tmp_funds.sort_values(ascending=False)
            tmp_funds = tmp_funds.dropna()
            pool_funds = tmp_funds.index[:fund_num]
            df_pos.loc[date, pool_funds] = 1.0

        df_pos = df_pos.div(df_pos.sum(1), 0)
        return df_pos

    def valid_test(self):

        indicator_df = pd.read_csv('data/fund_indicator/fund_indicator.csv', dtype={'fund_id': object}, index_col=['date'], parse_dates=['date'])
        indicator_df = indicator_df.set_index('fund_id', append=True)
        fund_nav = self.load_fund_nav(self.start_date, self.end_date)
        fund_nav = fund_nav.resample('m').last()
        fund_ret = fund_nav.pct_change(limit=1)

        df_res = {}
        dates = indicator_df.index.levels[0]
        for date, ndate in zip(dates[:-1], dates[1:]):
            ret = fund_ret.loc[ndate].dropna()
            df_res[ndate] = {}
            for indicator in indicator_df.columns:
                fi = indicator_df.loc[date, indicator].dropna()
                common_funds = np.intersect1d(ret.index, fi.index)
                ic = np.corrcoef(ret.loc[common_funds], fi.loc[common_funds])[1, 0]
                df_res[ndate][indicator] = ic
        df_res = pd.DataFrame.from_dict(df_res, orient='index')
        set_trace()
        return df_res

    def load_fund_pos(self, mz_id, date):

        df_pos = asset_mz_markowitz_pos.load('MZ.FI0050')
        fund_codes = base_ra_fund.load(globalids=df_pos.columns)
        fund_codes_dict = dict(zip(fund_codes.globalid.astype('str'), fund_codes.ra_code))
        df_pos = df_pos.rename(lambda x: fund_codes_dict[x], axis='columns')
        df_pos_new = df_pos.loc[date]
        df_pos_new = df_pos_new[df_pos_new > 0.0]
        set_trace()


def downstd(arr):
    arr = arr - arr.mean()
    ss = (arr[arr < 0]**2).sum()
    return ss / (len(arr) - 1)


if __name__ == '__main__':

    fi = FundIndicator('2010-01-01', '2018-09-01')
    fi.valid_test()
    # fi.cal_indicator()
    # fi.cal_indicator_index_pos('jensen_12')
    # fund_pos = fi.load_fund_pos(mz_id='MZ.FI0050', date='2018-08-31')


