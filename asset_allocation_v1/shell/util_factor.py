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
import util_fund


def load_factor():

    if os.path.exists('data/factor/factor_nav.csv'):
        factor_nav = pd.read_csv('data/factor/factor_nav.csv', index_col = ['date'], parse_dates =  ['date'])
        sdate = factor_nav.index[0]
        edate = factor_nav.index[-1]
        trade_dates = ATradeDate.trade_date(sdate, edate)
        factor_nav = factor_nav.reindex(trade_dates)
        return factor_nav

    factor_type = pd.read_csv('data/factor/factor_type.csv', encoding = 'gb2312')
    factor_type = factor_type[factor_type.state == 1]
    factor_index = caihui_tq_ix_basicinfo.find_index(factor_type.type)
    # factor_index.to_csv('data/factor/factor_name.csv',  index_label = 'type', encoding = 'gb2312')
    factor_code = factor_index.secode.values
    # 将国证A股指数加进去作为基准
    factor_code = np.append('2070000053', factor_code)
    factor_nav = caihui_tq_qt_index.load_multi_index_nav(factor_code)
    factor_nav.to_csv('data/factor/factor_nav.csv', index_label = 'date')

    return factor_nav


def load_fund():

    if os.path.exists('data/factor/fund_nav.csv'):
        fund_nav = pd.read_csv('data/factor/fund_nav.csv', index_col = ['date'], parse_dates = ['date'])
        return fund_nav

    pool_codes = list(base_ra_fund.find_type_fund(1).ra_code.ravel())
    fund_nav = base_ra_fund_nav.load_daily('2005-01-01', '2200-01-01', codes = pool_codes)
    fund_nav.to_csv('data/factor/fund_nav.csv', index_label = 'date')

    return fund_nav


def load_fund_names():

    df_fund = base_ra_fund.load()
    df_fund = df_fund.loc[:, ['ra_code', 'ra_name']]
    df_fund = df_fund.set_index('ra_code')
    fund_names_dict = df_fund.to_dict()['ra_name']

    return fund_names_dict


def load_factor_names():

    factor_name = pd.read_csv('data/factor/factor_name.csv', encoding = 'gb2312')
    factor_name = factor_name.loc[:, ['secode', 'name']]
    factor_name.secode = factor_name.secode.astype('str')
    factor_name = factor_name.set_index('secode')
    factor_name_dict = factor_name.to_dict()['name']

    return factor_name_dict


def load_init_factor():

    factor_name = pd.read_csv('data/factor/factor_name.csv', encoding = 'gb2312')
    init_factors = factor_name[factor_name.valid == 1].secode.astype('str').values

    return init_factors


def drop_dup(ret):

    corr = ret.corr()
    all_factors = sorted(corr.columns)
    factors = [all_factors[0]]
    for factor in all_factors[1:]:
        tmp_corr = corr.loc[factor, factors].max()
        if tmp_corr < 0.95:
            factors.append(factor)

    return ret.loc[:, factors]






