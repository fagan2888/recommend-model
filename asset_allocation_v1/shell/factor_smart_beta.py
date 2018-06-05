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
from util_optimize import ConstrRegression
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
    factor_index.to_csv('data/factor/factor_name.csv',  index_label = 'type', encoding = 'gb2312')
    factor_code = factor_index.secode.values
    factor_code = factor_code
    factor_nav = caihui_tq_qt_index.load_multi_index_nav(factor_code)
    factor_nav.to_csv('data/factor/factor_nav.csv', index_label = 'date')

    return factor_nav


def load_factor_nav():

    secodes = ['2070000060', '2070008092', '2070007967', '2070003741', '2070003742']
    factor_nav = pd.read_csv('data/factor/factor_nav.csv', index_col = ['date'], parse_dates =  ['date'])
    factor_nav = factor_nav.loc[:, secodes]
    factor_name = pd.read_csv('data/factor/factor_name.csv', encoding = 'gb2312')
    factor_name = factor_name[['secode', 'name']]
    factor_name.secode = factor_name.secode.astype('str')
    factor_name = factor_name.set_index('secode')
    factor_name = factor_name.to_dict()['name']
    factor_nav = factor_nav.rename(columns = factor_name)
    factor_nav.to_csv('data/factor/specific_factor_nav.csv', index_label = ['date'], encoding = 'gb2312')
    set_trace()


def load_fund():

    if os.path.exists('data/factor/fund_nav.csv'):
        fund_nav = pd.read_csv('data/factor/fund_nav.csv', index_col = ['date'], parse_dates = ['date'])
        return fund_nav

    pool_codes = list(base_ra_fund.find_type_fund(1).ra_code.ravel())
    fund_nav = base_ra_fund_nav.load_daily('2005-01-01', '2200-01-01', codes = pool_codes)
    fund_nav.to_csv('data/factor/fund_nav.csv', index_label = 'date')

    return fund_nav


def load_fund_all():

    path = 'data/factor/all_fund_nav.csv'
    if os.path.exists(path):
        fund_nav = pd.read_csv(path, index_col = ['date'], parse_dates = ['date'])
        return fund_nav

    pool_codes = list(base_ra_fund.find_type_fund(1).ra_code.ravel())
    pool_codes_other = list(base_ra_fund.find_type_fund(4).ra_code.ravel())
    pool_codes = pool_codes + pool_codes_other
    fund_nav = base_ra_fund_nav.load_daily('2005-01-01', '2200-01-01', codes = pool_codes)
    fund_nav.to_csv(path, index_label = 'date')

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


def find_smart_beta(factor_nav):

    factor_name = pd.read_csv('data/factor/factor_name.csv', encoding = 'gb2312')
    valid_secodes = factor_name[(factor_name.largecap == 1) & (factor_name.valid == 1)].secode.values
    set_trace()
    valid_secodes = factor_nav.columns.intersection(map(str, valid_secodes))
    factor_nav = factor_nav[valid_secodes]
    factor_nav = factor_nav.loc['2005':].dropna(1)
    factor_ret = factor_nav.pct_change().fillna(0.0)
    secodes = factor_ret.columns.values
    for secode in secodes:
        df_cmp = factor_ret[secode] - factor_ret['2070000060']
        print secode, len(df_cmp[df_cmp > 0]) / float(len(df_cmp))

    set_trace()


def find_smart_beta_small(factor_nav):

    factor_name = pd.read_csv('data/factor/factor_name.csv', encoding = 'gb2312')
    valid_secodes = factor_name[(factor_name.smallcap == 1) & (factor_name.valid == 1)].secode.values
    valid_secodes = factor_nav.columns.intersection(map(str, valid_secodes))
    factor_nav = factor_nav[valid_secodes]
    factor_nav = factor_nav.loc['2005':].dropna(1)
    factor_ret = factor_nav.pct_change().fillna(0.0)
    secodes = factor_ret.columns.values
    for secode in secodes:
        df_cmp = factor_ret[secode] - factor_ret['2070000187']
        print secode, len(df_cmp[df_cmp > 0]) / float(len(df_cmp))

    set_trace()


def smart_beta_fund_pool(factor_nav, fund_nav, sdate, edate):

    factor_ret = factor_nav.pct_change().loc[sdate:edate].dropna(1)
    fund_ret = fund_nav.pct_change().loc[sdate:edate].dropna(1)
    factor_name = pd.read_csv('data/factor/factor_name.csv', encoding = 'gb2312')
    fund_name_dict = load_fund_names()
    factor_name_dict = load_factor_names()

    hs300 = factor_nav['2070000060']
    zz500 = factor_nav['2070000187']
    df_mkt_values = pd.concat([hs300, zz500], 1)
    df_mkt_values.columns = ['largecap', 'smallcap']
    df_mkt_values = df_mkt_values.pct_change()
    df_mkt_values = df_mkt_values.loc[sdate:edate]
    fund_labels = fund_label_mv(fund_ret, df_mkt_values)

    fund_ret_small = fund_ret[fund_labels[fund_labels.mkt_label == 2.0].index]
    smallcap_beta = factor_name[(factor_name.smallcap == 1) & (factor_name.valid == 1) & (factor_name.pool == 1)].secode.values
    factor_sb_small = factor_ret[map(str, smallcap_beta)]
    fund_small_sb_labels = fund_label_sb(fund_ret_small, factor_sb_small)

    fund_small_sb_labels = fund_small_sb_labels.rename(columns = factor_name_dict, index = fund_name_dict)
    fund_small_sb_labels.to_csv('data/fund_pool/fund_small_sb_labels.csv', encoding = 'gb2312', index_label = 'fund_name')
    set_trace()


def fund_label_mv(fund_ret, df_mkt_values):

    x = df_mkt_values.values
    funds = fund_ret.columns
    df_label = pd.DataFrame(columns = ['rsquare', 'mkt_label'])
    for fund in funds:
        y = fund_ret[fund].values
        model = Lasso(alpha = 0, positive = True)
        res = model.fit(x, y)
        score = res.score(x, y)
        contrib = res.coef_/res.coef_.sum()
        if score < 0.8:
            label = 0
        elif contrib[0] > 0.8:
            label = 1
        elif contrib[1] > 0.8:
            label = 2
        else:
            label = 0

        df_label.loc[fund] = [score, label]

    return df_label


def fund_label_sb(fund_ret, factor_sb):

    x = factor_sb.values
    funds = fund_ret.columns
    # df_label = pd.DataFrame(columns = ['rsquare'] + ['sb_%d'%i for i in range(1, x.shape[1]+1)])
    df_label = pd.DataFrame(columns = np.append('rsquare', factor_sb.columns))
    for fund in funds:
        y = fund_ret[fund].values
        model = Lasso(alpha = 0, positive = True)
        res = model.fit(x, y)
        score = res.score(x, y)
        contrib = res.coef_/res.coef_.sum()
        df_label.loc[fund] = np.append(score, contrib)

    return df_label


def main():

    factor_nav = load_factor()
    fund_nav = load_fund()
    sdate = '2017-01-01'
    edate = '2018-01-01'

    # find_smart_beta(factor_nav)
    # find_smart_beta_small(factor_nav)
    smart_beta_fund_pool(factor_nav, fund_nav, sdate, edate)


if __name__ == '__main__':

    # load_factor()
    # load_fund()
    # load_factor_nav()
    # load_fund_names()
    # load_factor_names()
    main()






