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


    def cal_valid_date(self, asset):

        pool_id = asset_ra_pool.find_pool_id(asset)[0]
        pool_funds = asset_ra_pool_fund.load(pool_id)
        # dates = np.unique(pool_funds.index.get_level_values(0))
        dates = pool_funds.index.get_level_values(0).unique()
        # pre_valid = 1
        df_valid = pd.DataFrame(columns = ['valid', 'sh300_corr', 'zz500_corr', 'index_corr'])
        for date in dates:
            # funds = pool_funds.loc[dates[0]].index
            funds = pool_funds.loc[date].ra_fund_code.values
            df_fund_nav = self.df_nav_fund.loc[:, funds]
            # df_fund_nav = pd.DataFrame(fund_navs).dropna()
            df_fund_inc = df_fund_nav.pct_change().dropna()
            df_pool_inc = df_fund_inc.mean(1)
            df_pool_nav = (1 + df_pool_inc).cumprod()
            pool_dates = df_pool_nav.index

            base_corr_sh300 = np.corrcoef(df_pool_inc, self.base_incs.loc[pool_dates, '120000001'])[1, 0]
            base_corr_zz500 = np.corrcoef(df_pool_inc, self.base_incs.loc[pool_dates, '120000002'])[1, 0]
            index_corr = np.corrcoef(df_pool_inc, self.asset_incs.loc[pool_dates, asset])[1, 0]

            # base_corr_sh300 = np.polyfit(self.base_incs.loc[pool_dates, '120000001'], df_pool_inc, 1)[0]
            # base_corr_zz500 = np.polyfit(self.base_incs.loc[pool_dates, '120000002'], df_pool_inc, 1)[0]
            # index_corr = np.polyfit(self.asset_incs.loc[pool_dates, asset], df_pool_inc, 1)[0]
            if index_corr - max(base_corr_sh300, base_corr_zz500) > 0.03:
            # if (index_corr > 0.9) and (base_corr_sh300 < 0.8) and (base_corr_zz500 < 0.8):
                valid = 1
            else:
                valid = 0

            # if (this_valid == 1) and (pre_valid == 1):
            #     valid = 1
            # else:
            #     valid = 0

            # pre_valid = this_valid

            # print date, valid, round(base_corr_sh300, 2), round(base_corr_zz500, 2), round(index_corr, 2)
            df_valid.loc[date] = [valid, round(base_corr_sh300, 2), round(base_corr_zz500, 2), round(index_corr, 2)]

        df_valid['valid'] = relu(df_valid.valid.rolling(5).mean()-0.5)
        df_valid = df_valid.fillna(0.0)

        # print df_valid
        return df_valid


    def cal_valid_assets(self):
        df_valid_assets = {}
        for asset in self.factor_ids:
            df_valid_assets[asset] = self.cal_valid_date(asset).valid
        df_valid_assets = pd.DataFrame(df_valid_assets)

        return df_valid_assets


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


    def allocate_wave(self):
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

            data = {}
            for asset in assets:
                data[asset] = CommandMarkowitz.load_wavelet_nav_series(asset, end_date=date, reindex = self.trade_dates, wavelet_filter_num=2)
            df_nav = pd.DataFrame(data).fillna(method='pad')
            df_inc  = df_nav.pct_change().fillna(0.0)

            df_inc = df_inc.loc[:date]
            df_inc = df_inc.tail(26)
            if len(df_inc.columns) == 1:
                risks, returns, ws, sharpes = 0, 0, [1.0], 0
            else:
                risks, returns, ws, sharpes = PF.markowitz_r_spe(df_inc, bound)
            df_result.loc[date, assets] = np.array(ws).ravel()
            df_result.loc[date, ['sharpe', 'risk', 'return']] = [sharpes, risks, returns]
            df_result = df_result.fillna(0.0)
            # print df_result

        return df_result


    def allocate_avg(self):
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
            risks, returns, ws, sharpes = 0, 0, np.repeat(1.0/len(df_inc.columns), len(df_inc.columns)), 0
            df_result.loc[date, assets] = ws
            df_result.loc[date, ['sharpe', 'risk', 'return']] = [sharpes, risks, returns]
            df_result = df_result.fillna(0.0)
            # print df_result

        return df_result


    def handle(self):
        # df_valid_assets = self.valid_assets()
        self.allocate()


def relu(x):
    return np.sign(x)/2 + 0.5


if __name__ == '__main__':

    factor_ids = ['1200000%d'%i for i in range(52, 80)]
    # start_date = '2004-01-01'
    # start_date = '2010-06-01'
    start_date = '2005-06-01'
    end_date = '2018-04-24'
    vf = ValidFactor(factor_ids, start_date, end_date)
    vf.handle()
