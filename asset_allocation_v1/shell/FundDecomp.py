#!/home/yaojiahui/anaconda2/bin/python # coding=utf-8

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import click
import sys
sys.path.append('shell/')
from sklearn.linear_model import LinearRegression, Lasso
from scipy.stats import rankdata, spearmanr, pearsonr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import datetime
from ipdb import set_trace
import matplotlib
myfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=30)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')

from mk_test import mk_test
import Portfolio as PF
import Const
from db import asset_ra_pool_nav, asset_ra_pool_fund, asset_ra_pool, base_ra_fund_nav, base_ra_fund
import DBData
# from CommandMarkowitz import load_nav_series
import CommandMarkowitz


class FundDecomp(object):

    def __init__(self, factor_ids, start_date, end_date):

        self.factor_ids = factor_ids
        # self.base_ids = ['120000001', '120000002', '120000018']
        self.base_ids = ['120000001', '120000002']
        # self.other_ids = ['120000001', 'ERI000001', '120000014', 'ERI000002']
        self.other_ids = []
        self.market_ids = ['120000041']
        self.trade_dates = DBData.trade_dates(start_date, end_date)
        self.lookback = 52*2

        self.asset_navs, self.asset_incs = self.load_asset_nav(factor_ids, start_date, end_date)
        self.base_navs, self.base_incs = self.load_asset_nav(self.base_ids, start_date, end_date)
        self.other_navs, self.other_incs = self.load_asset_nav(self.other_ids, start_date, end_date)
        self.market_navs, self.market_incs = self.load_asset_nav(self.market_ids, start_date, end_date)
        # self.df_nav_fund = self.load_fund_nav()
        self.df_nav_fund = pd.read_csv('data/df_nav_fund.csv', index_col = ['td_date'], parse_dates = ['td_date'])
        fund_incs = self.df_nav_fund.pct_change()
        self.fund_incs = fund_incs.mask(abs(fund_incs) <= 0.001)


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
        # df_nav_fund.to_csv('data/df_nav_fund.csv', index_label = 'td_date')
        # fund_incs = df_nav_fund.pct_change()
        # fund_incs = fund_incs.mask(abs(fund_incs) <= 0.001)

        return df_nav_fund


    def load_valid_fund_nav(self):
        df_nav_fund = pd.read_csv('data/df_nav_fund.csv', index_col = ['td_date'], parse_dates = ['td_date'])
        fund_flag = pd.read_csv('data/fund_flag.csv')
        fund_flag.fund_code = fund_flag.fund_code.astype(str).str.zfill(6)
        valid_fund_codes = fund_flag[fund_flag.flag == 0.0].fund_code.values
        df_nav_fund = df_nav_fund.loc[:, valid_fund_codes]

        return df_nav_fund


    def cluster(self):
        dates = self.trade_dates
        df_nav_fund = self.load_valid_fund_nav()
        for i in range(self.lookback, len(dates)):
            start_date = dates[i-self.lookback]
            end_date = dates[i]
            tmp_fund_nav = df_nav_fund.loc[start_date:end_date]
            # tmp_basic_ret = self.base_incs.loc[start_date:end_date]
            tmp_fund_nav = tmp_fund_nav.dropna(1)
            tmp_fund_ret = tmp_fund_nav.pct_change().dropna()
            valid_fund_ret = tmp_fund_ret.mask(abs(tmp_fund_ret) <= 0.001)
            valid_fund_codes = []
            for fund in tmp_fund_ret.columns:
                fund_ret = tmp_fund_ret[fund]
                valid_ret = valid_fund_ret[fund].dropna()
                cond1 = len(fund_ret) - len(valid_ret) < 10
                if cond1:
                    valid_fund_codes.append(fund)
            tmp_fund_ret = tmp_fund_ret[valid_fund_codes]
            # tmp_fund_ret = tmp_fund_ret.merge(self.market_incs, left_index = True, right_index = True)
            # tmp_fund_alpha = pd.DataFrame(index = tmp_fund_ret.index)
            # y = tmp_fund_ret.iloc[:, -1].values
            # for fund in valid_fund_codes:
                # x = tmp_fund_ret[[fund]].values
                # res = sm.OLS(y, x).fit()
                # tmp_fund_alpha[fund] = res.resid
            tmp_market_ret = self.market_incs.loc[start_date:end_date]
            tmp_ind_ret = self.asset_incs.loc[start_date:end_date]
            self.train(tmp_fund_ret, tmp_market_ret, tmp_ind_ret)


    def train(self, ret, mret, iret):
        # 用四个行业因子['周期','金融','消费','成长']的超额收益来回归基金的超额收益
        # fund_alpha, fund_rsquare = self.cal_alpha(ret, mret)
        # ind_alpha, ind_rsquare = self.cal_alpha(iret, mret)

        assets = ret.columns
        dates = ret.index
        x = iret
        # scaler = StandardScaler()
        # x = scaler.fit_transform(ind_alpha)
        # x = ind_alpha.values
        # x = sm.add_constant(x)
        df_result = pd.DataFrame(columns = ['rsquare', 'largecap', 'smallcap', 'cycle', 'finance', 'consumption', 'growth'])
        for asset in assets:
            y = ret[[asset]].values
            # y = scaler.fit_transform(y)
            # res = sm.OLS(y,x).fit()
            # print res.summary(yname = asset, xname = ['constant', 'cycle', 'finance', 'consumption', 'growth'])
            res = Lasso(alpha = 0, fit_intercept = True, positive = True).fit(x, y)
            score = res.score(x,y)
            contrib = res.coef_/res.coef_.sum()
            score = np.round(score, 4)
            contrib = np.round(contrib, 4)
            df_result.loc[asset] = np.append(score, contrib)
            print asset, score, contrib
        df_fund = base_ra_fund.load()
        df_fund = df_fund.loc[:, ['ra_code', 'ra_name']]
        df_fund = df_fund.set_index('ra_code')
        df_result = pd.merge(df_fund, df_result, left_index = True, right_index = True, how = 'right')
        df_result.to_csv('data/factor_contrib.csv', index_label = 'fund_code', encoding = 'gbk')


    def cal_alpha(self, ret, mret):
        assets = ret.columns
        dates = ret.index
        df_alpha = pd.DataFrame(index = dates)
        df_rsquare = {}
        x = mret.values.ravel()
        # x = sm.add_constant(x)
        for asset in assets:
            y = ret[asset].values
            # res = sm.OLS(y,x).fit()
            # df_alpha[asset] = res.resid
            # df_rsquare[asset] = res.rsquared
            # df_alpha[asset] = y - x
            df_alpha[asset] = y

        df_rsquare = pd.Series(df_rsquare)

        return df_alpha, df_rsquare


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
        self.cluster()
        # self.generate_black_list()
        # self.load_valid_fund_nav()


def plot_fund_nav(fund_codes, start_date, end_date, name = None, title = None):
    df_nav_fund = pd.read_csv('data/df_nav_fund.csv', index_col = ['td_date'], parse_dates = ['td_date'])
    df_nav_fund = df_nav_fund.loc[start_date:end_date, fund_codes]
    df_nav_fund = df_nav_fund/df_nav_fund.iloc[0]
    if len(df_nav_fund.columns) > 10:
        df_nav_fund = df_nav_fund.sample(10, axis = 1)
    # df_nav_fund.plot(figsize = (30, 20))
    df_fund = base_ra_fund.load()
    fund_names = df_fund[df_fund.ra_code.isin(df_nav_fund.columns)].ra_name
    df_nav_fund.columns = fund_names
    df_nav_fund.plot(figsize = (35, 20))
    # fig = plt.figure(figsize = (30, 20))
    # ax = fig.add_subplot(111)
    # ax.plot(df_nav_fund, label = fund_names)
    plt.legend(prop = myfont)
    if title is not None:
        plt.title(title, fontdict = {'fontsize': 30})
    if name is not None:
        plt.savefig('/home/yaojiahui/Desktop/'+name+'.png')
    else:
        plt.savefig('/home/yaojiahui/Desktop/tmp.png')

    # plt.show()


if __name__ == '__main__':

    # factor_ids = ['1200000%d'%i for i in range(52, 80)]
    # ['周期','金融','消费','成长']
    factor_ids = ['120000001', '120000002', '120000055', '120000056', '120000058', '120000078']
    start_date = '2016-01-01'
    end_date = '2018-05-01'
    fd = FundDecomp(factor_ids, start_date, end_date)
    fd.handle()
