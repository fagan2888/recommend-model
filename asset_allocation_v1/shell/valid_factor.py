#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('shell/')
from ipdb import set_trace
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.decomposition import PCA

from db import *
from trade_date import ATradeDate
# from util_optimize import ConstrRegression
from scipy.stats import rankdata, ttest_rel
import util_fund
import util_factor
from cluster_tree import clusterKMeansBase

class ValidFactor(object):


    def __init__(self, sdate = None, edate = None):

        self.sdate = sdate
        self.edate = edate
        self.factor_names = util_factor.load_factor_names()
        self.fund_names = util_factor.load_fund_names()
        self.init_factors = util_factor.load_init_factor()
        self.factor_nav = util_factor.load_factor()
        self.fund_nav = util_factor.load_fund()

        self.find_valid_factor('2011-01-01')
        self.find_valid_large()
        self.find_valid_small()

    def find_valid_factor(self, edate):

        period = 66
        init_nav = self.factor_nav.loc[:, self.init_factors]
        init_nav = init_nav.loc['2005-02-03':edate].dropna(1)
        init_ret = init_nav.pct_change(period).dropna()
        init_ret = init_ret.replace(0.0, np.nan).dropna(1)
        # base_ret = init_ret['2070000053'].values
        # init_ret_div = init_ret.sub(base_ret, 0)

        df_test = pd.DataFrame(columns = ['stats', 'pvalue'])
        for factor in init_ret.columns:
            res = ttest_rel(init_ret[factor], init_ret['2070000053'])
            df_test.loc[factor] = res

        df_test = df_test[df_test.stats > 0][df_test.pvalue < 0.05]
        df_test = df_test.sort_values('stats', ascending = False)
        valid_factors = df_test.index.values
        valid_ret = init_ret.loc[:, valid_factors]
        valid_corr = valid_ret.corr()
        _,cluster_res,_ = clusterKMeansBase(valid_corr,maxNumClusters=10,n_init=10)

        layer_res = {}
        valid_ret_0 = init_ret.loc[:, cluster_res[0]]
        valid_ret_1 = init_ret.loc[:, cluster_res[1]]
        corr0_sh300 = np.corrcoef(valid_ret_0.mean(1), init_ret['2070000060'])[1,0]
        corr1_sh300 = np.corrcoef(valid_ret_1.mean(1), init_ret['2070000060'])[1,0]
        if corr0_sh300 > corr1_sh300:
            layer_res[0] = cluster_res[0]
            layer_res[1] = cluster_res[1]
        elif corr0_sh300 <= corr1_sh300:
            layer_res[0] = cluster_res[1]
            layer_res[1] = cluster_res[0]

        layer_ret_0 = init_ret.loc[:, layer_res[0]]
        layer_ret_1 = init_ret.loc[:, layer_res[1]]

        # corr0 = layer_ret_0.corr()
        # corr1 = layer_ret_1.corr()
        # corr0 = corr0.rename(columns = self.factor_names, index = self.factor_names)
        # corr1 = corr1.rename(columns = self.factor_names, index = self.factor_names)
        # corr0.to_csv('data/factor/corr0.csv', encoding = 'gb2312')
        # corr1.to_csv('data/factor/corr1.csv', encoding = 'gb2312')

        self.large_ret = layer_ret_0
        self.small_ret = layer_ret_1
        self.init_ret = init_ret


    def find_valid_large(self):

        df_test = pd.DataFrame(columns = ['stats', 'pvalue'])
        for factor in self.large_ret.columns:
            res = ttest_rel(self.large_ret[factor], self.init_ret['2070000060'])
            df_test.loc[factor] = res

        df_test = df_test[df_test.stats > 0][df_test.pvalue < 0.05]
        df_test = df_test.sort_values('stats', ascending = False)
        valid_large_ret = self.init_ret.loc[:, df_test.index]
        self.valid_large_ret = util_factor.drop_dup(valid_large_ret)


        # self.valid_large_ret.corr().to_csv('data/factor/corr_large.csv')
        # tmp_ret = pd.concat([self.factor_nav['2070000060'], self.factor_nav.loc[:, self.valid_large_ret.columns]], 1)
        # tmp_ret = tmp_ret.rename(columns = self.factor_names)

        # tmp_ret_train = tmp_ret.loc['2005-02-03':'2011-01-01']
        # tmp_ret_train = tmp_ret_train / tmp_ret_train.iloc[0]
        # tmp_ret_train.to_csv('data/factor/valid_factor_large_train.csv', encoding = 'gb2312', index_label = 'date')

        # tmp_ret_test = tmp_ret.loc['2011-01-01':]
        # tmp_ret_test = tmp_ret_test / tmp_ret_test.iloc[0]
        # tmp_ret_test.to_csv('data/factor/valid_factor_large_test.csv', encoding = 'gb2312', index_label = 'date')


    def find_valid_small(self):

        df_test = pd.DataFrame(columns = ['stats', 'pvalue'])
        for factor in self.small_ret.columns:
            res = ttest_rel(self.small_ret[factor], self.init_ret['2070000187'])
            df_test.loc[factor] = res

        df_test = df_test[df_test.stats > 0][df_test.pvalue < 0.05]
        df_test = df_test.sort_values('stats', ascending = False)
        valid_small_ret = self.init_ret.loc[:, df_test.index]
        self.valid_small_ret = util_factor.drop_dup(valid_small_ret)


        # self.valid_large_ret.corr().to_csv('data/factor/corr_small.csv')
        # tmp_ret = pd.concat([self.factor_nav['2070000187'], self.factor_nav.loc[:, self.valid_small_ret.columns]], 1)
        # tmp_ret = tmp_ret.rename(columns = self.factor_names)

        # tmp_ret_train = tmp_ret.loc['2005-02-03':'2011-01-01']
        # tmp_ret_train = tmp_ret_train / tmp_ret_train.iloc[0]
        # tmp_ret_train.to_csv('data/factor/valid_factor_small_train.csv', encoding = 'gb2312', index_label = 'date')

        # tmp_ret_test = tmp_ret.loc['2011-01-01':]
        # tmp_ret_test = tmp_ret_test / tmp_ret_test.iloc[0]
        # tmp_ret_test.to_csv('data/factor/valid_factor_small_test.csv', encoding = 'gb2312', index_label = 'date')


    def fund_div(self, day):

        fund_ret = self.fund_nav.pct_change()
        fund_ret = fund_ret.loc[:day].iloc[-252:]
        fund_ret = fund_ret.replace(0.0, np.nan)
        fund_ret = fund_ret.dropna(axis = 1, thresh = 10)
        fund_ret = fund_ret.fillna(0.0)

        large_ret = self.factor_nav[['2070000060']].pct_change()
        large_ret = large_ret.loc[:day].iloc[-252:]

        small_ret = self.factor_nav[['2070000187']].pct_change()
        small_ret = small_ret.loc[:day].iloc[-252:]

        funds = fund_ret.columns.values
        df_mv = pd.concat([large_ret, small_ret], 1)
        x = df_mv.values
        df_res = pd.DataFrame(columns = ['rsquare', 'large', 'small'])
        for fund in funds:

            y = fund_ret[fund].values

            mod = Lasso(alpha = 0, fit_intercept = True, positive = True)
            res = mod.fit(x, y)
            score = res.score(x, y)
            contrib = res.coef_/res.coef_.sum()

            score = np.round(score, 4)
            contrib = np.round(contrib, 4)
            df_res.loc[fund] = np.append(score, contrib)

        large_funds = df_res[df_res.rsquare > 0.5][df_res.large > 0.7]
        small_funds = df_res[df_res.rsquare > 0.5][df_res.small > 0.7]

        self.large_funds_ret = fund_ret.loc[:, large_funds.index]
        self.small_funds_ret = fund_ret.loc[:, small_funds.index]


    def cal_fund_pool_large(self, day):

        # fund_ret = self.fund_nav.pct_change()
        # fund_ret = fund_ret.loc[:day].iloc[-252:]
        # fund_ret = fund_ret.replace(0.0, np.nan)
        # fund_ret = fund_ret.dropna(axis = 1, thresh = 10)
        # fund_ret = fund_ret.fillna(0.0)


        # 
        ## Cal LargeCap FundPool
        #
        large_ret = self.factor_nav.pct_change().loc[:day, self.valid_large_ret.columns].iloc[-252:]
        funds = self.large_funds_ret.columns.values
        df_res_large = pd.DataFrame(columns = np.append(['rsquare', 'alpha'], large_ret.columns))
        x = large_ret.values
        for fund in funds:

            y = self.large_funds_ret[fund].values
            mod = Lasso(alpha = 0, fit_intercept = True, positive = True)
            res = mod.fit(x, y)
            score = res.score(x, y)
            alpha = res.intercept_
            contrib = res.coef_/res.coef_.sum()

            score = np.round(score, 4)
            alpha = np.round(alpha, 4)
            contrib = np.round(contrib, 4)
            df_res_large.loc[fund] = np.append([score, alpha], contrib)

        # df_res_large = df_res_large.rename(columns = self.factor_names, index = self.fund_names)
        # df_res_large.to_csv('data/fund_pool/large_fund_pool.csv', index_label = 'fund_code', encoding = 'gbk')
        # fund_codes = []
        df_res_large = df_res_large[df_res_large.rsquare > 0.8]

        # for factor in self.valid_large_ret.columns:
        #     tmp_fund_codes = df_res_large[df_res_large[factor] > 0.8].index
        #     fund_codes = np.append(fund_codes, tmp_fund_codes)
        df_res_large = df_res_large.sort_values('alpha', ascending = False)
        fund_codes = df_res_large.index.values[:20]

        self.large_fund_codes = fund_codes


    def cal_fund_pool_small(self, day):

        #
        ## Cal SmallCap FundPool
        #
        small_ret = self.factor_nav.pct_change().loc[:day, self.valid_small_ret.columns].iloc[-252:]
        funds = self.small_funds_ret.columns.values
        df_res_small = pd.DataFrame(columns = np.append(['rsquare', 'alpha'], small_ret.columns))
        x = small_ret.values
        for fund in funds:

            y = self.small_funds_ret[fund].values
            mod = Lasso(alpha = 0, fit_intercept = True, positive = True)
            res = mod.fit(x, y)
            score = res.score(x, y)
            alpha = res.intercept_
            contrib = res.coef_/res.coef_.sum()

            score = np.round(score, 4)
            alpha = np.round(alpha, 4)
            contrib = np.round(contrib, 4)
            df_res_small.loc[fund] = np.append([score, alpha], contrib)

        # df_res_small = df_res_small.rename(columns = self.factor_names, index = self.fund_names)
        # df_res_small.to_csv('data/fund_pool/small_fund_pool.csv', index_label = 'fund_code', encoding = 'gbk')

        # fund_codes = []
        df_res_small = df_res_small[df_res_small.rsquare > 0.8]
        # for factor in self.valid_small_ret.columns:
        #     tmp_fund_codes = df_res_small[df_res_small[factor] > 0.8].index
        #     fund_codes = np.append(fund_codes, tmp_fund_codes)

        df_res_small = df_res_small.sort_values('alpha', ascending = False)
        fund_codes = df_res_small.index.values[:20]

        self.small_fund_codes = fund_codes


    def handle(self, edate, method):

        # self.find_valid_factor(edate)
        # self.find_valid_factor('2011-01-01')
        # self.find_valid_large()
        # self.find_valid_small()
        self.fund_div(edate)

        if method == 'large':
            self.cal_fund_pool_large(edate)
        if method == 'small':
            self.cal_fund_pool_small(edate)


if __name__ == '__main__':

    sdate = '2010-01-01'
    edate = '2018-05-01'
    vf = ValidFactor(sdate, edate)
    vf.handle('2018-01-01', 'large')






