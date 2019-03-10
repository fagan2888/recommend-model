#coding=utf-8
'''
Created on: Mar. 6, 2019
Modified on: Mar. 11, 2019
Author: Shixun Su, Boyang Zhou
Contact: sushixun@licaimofang.com
'''

import sys
import logging
import click
import warnings
# import multiprocessing
# from multiprocessing import Pooli, Manager
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, gaussian_kde
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
# import tensorflow as tf
sys.path.append('shell')
from db import caihui_tq_ix_comp, caihui_tq_sk_basicinfo, caihui_tq_sk_dquoteindic, caihui_tq_sk_finindic
from trade_date import ATradeDate
from util_timestamp import *
from ipdb import set_trace


logger = logging.getLogger(__name__)



class MultiFators(object):

    def __init__(self, index_id='2070000191', begin_date='20000101'):

        self.__stock_pool = caihui_tq_ix_comp.load_index_constituents(index_id)
        # df_CSI800_constituents = pd.read_csv('data/CSI800_constituents.csv')
        # df_CSI800_pool = caihui_tq_sk_basicinfo.load_stock_code_info(stock_codes=df_CSI800_constituents['Constituents Code'], current=False)

        self._df_stock_nav = caihui_tq_sk_dquoteindic.load_stock_nav(begin_date=begin_date, stock_ids=self.__stock_pool.index)
        self._df_stock_ret = self._df_stock_nav.pct_change().iloc[1:]

        # self._df_stock_market_data = caihui_tq_sk_dquoteindic.load_stock_market_data(begin_date=begin_date, stock_ids=self.__stock_pool.index)
        # self._df_stock_financial_data = caihui_tq_sk_finindic.load_stock_financial_data(begin_date=begin_date, stock_ids=self.__stock_pool.index)
        # self._df_stock_industry = caihui_tq_sk_basicinfo.load_stock_industry(stock_ids=self.__stock_pool.index)
        self._df_stock_indicators = self._cal_indicators()

    @property
    def stock_pool(self):
        return self.__stock_pool

    def _cal_indicators(self, window=120):

        df = pd.DataFrame()
        df['volatility'] = self._df_stock_ret.rolling(window=window).std().iloc[window:].unstack().dropna()

        return df

    def _cal_weight(self, factor, method='daoshu'):

        if method == 'daoshu':

            factor = factor.unstack().T
            weight = factor.applymap(lambda x: 1/x if x>0.0 else np.nan)
            weight = weight.apply(lambda x: x/x.sum(), axis='columns')
            set_trace()

    def cal_ret_list(self):

        df_weight = self._cal_weight(self._df_stock_indicators.volatility)
        ser_portfolio_ret = (df_weight * self._df_stock_ret).sum(axis='columns')

        return ser_portfolio_ret

    def analysis(self, ser_portfolio_ret):

        ret = (ser_portfolio_ret + 1).pi - 1
        risk = ser_portfolio_ret.std()
        sharpe_ratio = ret / risk

        print(ret, risk, sharpe_ratio)


    def GLS(self, Y, X):

        index = X.index.intersection(Y.index)
        X = X.reindex(index)
        Y = Y.reindex(index)

        GLS = sm.GLS(Y, X)
        GLS_results = GLS.fit()
        print(GLS_results.summary())

        return GLS_results




# from db import caihui_tq_qt_index
# Ret_indexing = caihui_tq_qt_index.load_index_nav(begin_date='20100101', index_ids=['2070000191']).pct_change().iloc[1:]
# df_CSI800_pool = caihui_tq_ix_comp.load_index_constituents('2070000191')
# PS_ratio = caihui_tq_sk_finindic.load_stock_financial_data(begin_date='20100101', stock_ids=df_CSI800_pool.index)
# PS_ratio = PS_ratio.ps_lfy.unstack().T.fillna(0.0).iloc[1:]

# GLS_TEST = GLS(Ret_indexing, PS_ratio)
# GLS_TEST_params = GLS_TEST._results.params
# GLS_TEST_resid = GLS_TEST._results.resid
# GLS_TEST_params.mean()

# GLS(Ret_indexing,PS_ratio.mean(axis=1))

if __name__ == '__main__':

    mf = MultiFators(begin_date='20160101')
    x = mf.cal_ret_list()

