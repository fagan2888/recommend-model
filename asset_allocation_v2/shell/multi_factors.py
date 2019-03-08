#coding=utf-8
'''
Created on: Mar. 6, 2019
Author: Shixun Su, Boyang Zhou
Contact: sushixun@licaimofang.com
'''

import sys
import logging
import click
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, gaussian_kde
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
sys.path.append('shell')
from db import caihui_tq_ix_comp, caihui_tq_sk_basicinfo, caihui_tq_sk_dquoteindic, caihui_tq_sk_finindic
from trade_date import ATradeDate
from util_timestamp import *
from ipdb import set_trace


logger = logging.getLogger(__name__)


def load_CSI800():

    # df_CSI800_pool = caihui_tq_ix_comp.load_index_constituents('2070000191')
    df_CSI800_constituents = pd.read_csv('data/CSI800_constituents.csv')
    df_CSI800_pool = caihui_tq_sk_basicinfo.load_stock_code_info(stock_codes=df_CSI800_constituents['Constituents Code'], current=False)

    df_CSI800_market_data = caihui_tq_sk_dquoteindic.load_stock_market_data(begin_date='20000101', stock_ids=df_CSI800_pool.index)
    df_CSI800_financial_data = caihui_tq_sk_finindic.load_stock_financial_data(begin_date='20000101', stock_ids=df_CSI800_pool.index)
    df_CSI800_industry = caihui_tq_sk_basicinfo.load_stock_industry(stock_ids=df_CSI800_pool.index)
    set_trace()
    result = pd.merge(df_CSI800_dquote_indic, df_CSI800_financial_data, how='outer', on=['stock_id', 'date'])

    set_trace()

def GLS(Y, X):

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

    load_CSI800()

