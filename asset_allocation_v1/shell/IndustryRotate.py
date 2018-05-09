#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import click
import sys
sys.path.append('shell/')
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata, spearmanr
import datetime
from ipdb import set_trace

from mk_test import mk_test
import Portfolio as PF
import Const
from db import asset_ra_pool_nav, asset_ra_pool_fund, asset_ra_pool, base_ra_fund_nav, base_ra_fund
import DBData
# from CommandMarkowitz import load_nav_series
import CommandMarkowitz


class ValidFactor(object):

    def __init__(self, factor_ids, start_date, end_date):

        self.factor_ids = factor_ids
        # self.base_ids = ['120000001', '120000002', '120000018']
        self.base_ids = ['120000001', '120000002']
        # self.other_ids = ['120000001', 'ERI000001', '120000014', 'ERI000002']
        self.other_ids = []
        self.trade_dates = DBData.trade_dates(start_date, end_date)

        self.asset_navs, self.asset_incs = self.load_asset_nav(factor_ids, start_date, end_date)
        self.base_navs, self.base_incs = self.load_asset_nav(self.base_ids, start_date, end_date)
        self.other_navs, self.other_incs = self.load_asset_nav(self.other_ids, start_date, end_date)
        # self.df_nav_fund = self.load_fund_nav()
        self.df_nav_fund = pd.read_csv('data/df_nav_fund.csv', index_col = ['td_date'], parse_dates = ['td_date'])


    def load_asset_nav(self, factor_ids, start_date, end_date):

        trade_dates = DBData.trade_dates(start_date, end_date)
        asset_navs = {}
        for factor_id in factor_ids:
            asset_navs[factor_id] = CommandMarkowitz.load_nav_series(factor_id, reindex = trade_dates)

        df_asset_navs = pd.DataFrame(asset_navs)
        df_asset_incs = df_asset_navs.pct_change().dropna()

        return df_asset_navs, df_asset_incs


    def load_fund_nav(self):
        pool_codes = list(base_ra_fund.find_type_fund(1).ra_code.ravel())
        df_nav_fund  = base_ra_fund_nav.load_daily(start_date, end_date, codes = pool_codes)
        df_nav_fund = df_nav_fund.loc[self.trade_dates]
        df_nav_fund.to_csv('data/df_nav_fund.csv', index_label = 'td_date')

        return df_nav_fund


    def cal_rotate(self):
        asset_ret = self.asset_navs.pct_change(52).dropna()
        asset_rank = asset_ret.apply(rankdata, 1)
        asset_rank.columns = ['钢铁','银行','食品','计算机']
        asset_rank.to_csv('data/industry_rotate.csv', index_label = 'date', encoding = 'gbk')
        set_trace()


    def allocate(self):
        dates = self.trade_dates[self.trade_dates >= datetime.datetime(2012, 7, 27).date()]
        # dates = self.trade_dates[self.trade_dates >= datetime.datetime(2018, 3, 1).date()]
        asset_incs = pd.concat([self.asset_incs, self.other_incs], 1)
        # df_valid_assets = self.cal_valid_assets()
        df_valid_assets = pd.read_csv('data/df_valid_assets.csv', index_col = ['date'], parse_dates = ['date'])
        bound = []
        for asset in asset_incs.columns:
            bound.append(Const.bound[asset])

        df_result = pd.DataFrame(columns = np.append(asset_incs.columns.values, ['sharpe', 'risk', 'return']))
        for date in dates:
            print date
            valid_assets = df_valid_assets[df_valid_assets.index < date.strftime('%Y-%m-%d')].tail(1)
            valid_assets = valid_assets[valid_assets == 1].dropna(1).columns.values
            assets = np.append(valid_assets, self.other_ids)
            df_inc = asset_incs.loc[:date, assets]
            df_inc = df_inc.tail(26)
            risks, returns, ws, sharpes = PF.markowitz_bootstrape(df_inc, bound, 36)
            df_result.loc[date, assets] = ws
            df_result.loc[date, ['sharpe', 'risk', 'return']] = [sharpes, risks, returns]
            df_result = df_result.fillna(0.0)
            # print df_result

        return df_result


    def handle(self):
        self.cal_rotate()


if __name__ == '__main__':

    # factor_ids = ['1200000%d'%i for i in range(52, 80)]
    factor_ids = ['120000055', '120000056', '120000058', '120000078']
    start_date = '2005-06-01'
    end_date = '2018-05-01'
    vf = ValidFactor(factor_ids, start_date, end_date)
    vf.handle()
