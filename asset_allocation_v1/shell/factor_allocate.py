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


def factor_ortho(factor_nav, sdate, edate):

    n_components = 10
    factor_nav = factor_nav.loc[sdate:edate].dropna(1)
    reindex = ATradeDate.trade_date(sdate, edate)
    factor_ret = factor_nav.reindex(reindex).fillna(method = 'pad').pct_change().dropna()

    x = factor_ret.values
    pca = PCA(n_components = n_components)
    pca.fit(x)
    ortho_factor = pca.transform(x)
    df = pd.DataFrame(data = ortho_factor, index = factor_ret.index, columns = ['pca_%d'%i for i in range(1, n_components+1)])
    df = (1 + df).cumprod()
    df.to_csv('data/factor/ortho_factor.csv', index_label = 'date')

    # ret = factor_ret.values.T
    # sigma = np.cov(ret)
    # s,v,d = np.linalg.svd(sigma)
    # ortho_factor = np.dot(d, ret).T
    # ortho_factor = ortho_factor[:, :n_components]
    # df = pd.DataFrame(data = ortho_factor, index = factor_ret.index, columns = ['pca_%d'%i for i in range(1, n_components+1)])
    set_trace()


def ortho_factor_ret(factor_nav, sdate, edate):

    n_components = 10
    factor_nav = factor_nav.loc[sdate:edate].dropna(1)
    reindex = ATradeDate.trade_date(sdate, edate)
    factor_ret = factor_nav.reindex(reindex).fillna(method = 'pad').pct_change().dropna()

    ret = factor_ret.values.T
    sigma = np.cov(ret)
    s,v,d = np.linalg.svd(sigma)
    a = np.diag(1/d.sum(1))
    ortho_factor = np.dot(d, ret)
    # ortho_factor = ortho_factor.cumsum(0)
    factor_ret = np.dot(a, ortho_factor).T
    factor_ret = factor_ret[:, :n_components]

    weight = (v / v.sum())[:n_components]
    scale = 1 / weight.cumsum()
    factor_ret = np.multiply(factor_ret, weight)
    factor_ret = factor_ret.cumsum(1)
    factor_ret = np.multiply(factor_ret, scale)


    # pca_columns = ['pca_%d'%i for i in range(1, n_components+1)]
    # df = pd.DataFrame(data = factor_ret, index = factor_ret.index, columns = pca_columns)
    # df = (1 + df).cumprod()
    # df.to_csv('data/factor/factor_ret.csv')

    return factor_ret


def ortho_fund_ret(factor_nav, sdate, edate):

    n_components = 10
    factor_nav = factor_nav.loc[sdate:edate].dropna(1)
    reindex = ATradeDate.trade_date(sdate, edate)
    factor_ret = factor_nav.reindex(reindex).fillna(method = 'pad').pct_change().dropna()
    reindex = factor_ret.index

    ret = factor_ret.values.T
    sigma = np.cov(ret)
    s,v,d = np.linalg.svd(sigma)
    a = np.diag(1/d.sum(1))
    ortho_factor = np.dot(d, ret)
    # ortho_factor = ortho_factor.cumsum(0)
    factor_ret = np.dot(a, ortho_factor).T
    factor_ret = factor_ret[:, :n_components]

    # weight = (v / v.sum())[:n_components]
    # scale = 1 / weight.cumsum()
    # factor_ret = np.multiply(factor_ret, weight)
    # factor_ret = factor_ret.cumsum(1)
    # factor_ret = np.multiply(factor_ret, scale)


    pca_columns = ['pca_%d'%i for i in range(1, n_components+1)]
    df = pd.DataFrame(data = factor_ret, index = reindex, columns = pca_columns)
    df = (1 + df).cumprod()
    df.to_csv('data/factor/ortho_fund_ret.csv')

    return factor_ret


def pca_ret(ret, n_components):

    ret = ret.T
    sigma = np.cov(ret)
    s,v,d = np.linalg.svd(sigma)
    a = np.diag(1/d.sum(1))
    ortho_factor = np.dot(d, ret)
    factor_ret = np.dot(a, ortho_factor).T
    factor_ret = factor_ret[:, :n_components]

    # weight = (v / v.sum())[:n_components]
    # factor_ret = np.multiply(factor_ret, weight)

    # wa_ret = np.zeros_like(factor_ret)
    # for i in range(n_components):
    #     wa_ret[:, i] = (factor_ret[:, 0]*weight[0] + factor_ret[:, i]*weight[i])/(weight[0] + weight[i])

    return factor_ret


def factor_combine(ret):

    ret = ret.T
    sigma = np.cov(ret)
    s,v,d = np.linalg.svd(sigma)
    a = np.diag(1/d.sum(1))
    ortho_factor = np.dot(d, ret)
    set_trace()
    ortho_factor_ret = np.dot(a, ortho_factor).T
    ortho_factor_ret = ortho_factor_ret[:, 0]


def ortho_factor_recognize(factor_nav, sdate, edate):

    n_components = 10
    factor_nav = factor_nav.loc[sdate:edate].dropna(1)
    reindex = ATradeDate.trade_date(sdate, edate)
    factor_ret = factor_nav.reindex(reindex).fillna(method = 'pad').pct_change().dropna()

    x = factor_ret.values

    # pca = PCA(n_components = n_components)
    # pca.fit(x)
    # ortho_factor = pca.transform(x)
    ortho_factor = pca_ret(x, 10)

    pca_columns = ['pca_%d'%i for i in range(1, n_components+1)]
    df = pd.DataFrame(data = ortho_factor, index = factor_ret.index, columns = pca_columns)
    df_pca_factor = pd.concat([df, factor_ret], 1)
    corr = df_pca_factor.corr()
    secodes = []
    corrs = []
    for pca in pca_columns:
        print pca,
        corr_pca = corr.loc[pca].dropna()
        corr_pca = corr_pca.iloc[n_components:]
        corr_sort = corr_pca.sort_values(ascending = False)
        corr_sort.index = caihui_tq_ix_basicinfo.load_index_name(corr_sort.index).name
        corr_sort.to_csv('data/pca_recognize/%s_corr.csv'%pca, index_label = 'name', encoding = 'gb2312')
        # print pca
        # print corr_sort
        secodes.append(corr_sort.index[-1])
        corrs.append(corr_sort.values[-1])
    factor_name = caihui_tq_ix_basicinfo.load_index_name(secodes)
    factor_name['corr'] = corrs
    set_trace()


def ortho_fund_recognize(fund_nav, factor_nav, sdate, edate):

    n_components = 10
    factor_nav = factor_nav.loc[sdate:edate].dropna(1)
    fund_nav = fund_nav.loc[sdate:edate].dropna(1)
    reindex = ATradeDate.trade_date(sdate, edate)
    factor_ret = factor_nav.reindex(reindex).fillna(method = 'pad').pct_change().dropna()
    fund_ret = fund_nav.reindex(reindex).fillna(method = 'pad').pct_change().dropna()

    x = fund_ret.values

    # pca = PCA(n_components = n_components)
    # pca.fit(x)
    # ortho_factor = pca.transform(x)
    ortho_factor = pca_ret(x, n_components)

    pca_columns = ['pca_%d'%i for i in range(1, n_components+1)]
    df = pd.DataFrame(data = ortho_factor, index = factor_ret.index, columns = pca_columns)
    df_pca_factor = pd.concat([df, factor_ret], 1)
    corr = df_pca_factor.corr()
    secodes = []
    corrs = []
    for pca in pca_columns:
        print pca,
        corr_pca = corr.loc[pca].dropna()
        corr_pca = corr_pca.iloc[n_components:]
        corr_sort = corr_pca.sort_values(ascending = False)
        corr_sort.index = caihui_tq_ix_basicinfo.load_index_name(corr_sort.index).name
        corr_sort.to_csv('data/pca_recognize_fund/%s_corr.csv'%pca, index_label = 'name', encoding = 'gb2312')
        # print pca
        # print corr_sort
        secodes.append(corr_sort.index[-1])
        corrs.append(corr_sort.values[-1])
    factor_name = caihui_tq_ix_basicinfo.load_index_name(secodes)
    factor_name['corr'] = corrs
    set_trace()


def fund_ret_attrib(factor_nav, fund_nav, sdate, edate):

    factor_nav = factor_nav.loc[sdate:edate].dropna(1)
    fund_nav = fund_nav.loc[sdate:edate].dropna(1)
    fund_nav = util_fund.tshare_filter(fund_nav)
    # reindex = ATradeDate.week_trade_date(sdate, edate)
    reindex = ATradeDate.trade_date(sdate, edate)

    factor_ret = factor_nav.reindex(reindex).fillna(method = 'pad').pct_change().dropna()
    fund_ret = fund_nav.reindex(reindex).fillna(method = 'pad').pct_change().dropna()

    n_components = 5
    x = factor_ret.values
    pca = PCA(n_components = n_components)
    pca.fit(x)
    x = pca.transform(x)
    set_trace()

    # df = pd.DataFrame(data = x, index = factor_ret.index, columns = ['pca_%d'%i for i in range(1, n_components+1)])
    # df = (1 + df).cumprod()
    # df.to_csv('data/factor/ortho_factor.csv', index_label = 'date')

    # factor_name = caihui_tq_ix_basicinfo.load_index_name(factor_ret.columns)
    # df_result = pd.DataFrame(columns = np.append('score', factor_name.name))
    df_result = pd.DataFrame(columns = np.append('score', ['pca_%d'%i for i in range(1, n_components+1)]))
    df_weight = pd.DataFrame(columns = np.append('weight', ['pca_%d'%i for i in range(1, n_components+1)]))
    count = 0

    for fund in fund_ret.columns:
        y = fund_ret.loc[:, fund].values

        mod = ConstrRegression()
        mod.fit(x, y)
        score = mod.score(x, y)
        contrib = mod.contrib
        weight = mod.w

        # contrib = mod.res.x
        # contrib = mod.res.x/mod.res.x.sum()

        # mod = Lasso(alpha = 0, fit_intercept = True, positive = True)
        # mod = LinearRegression(fit_intercept = False)
        # res = mod.fit(x, y)
        # score = res.score(x, y)
        # contrib = res.coef_/res.coef_.sum()

        score = np.round(score, 4)
        contrib = np.round(contrib, 4)
        df_result.loc[fund] = np.append(score, contrib)
        df_weight.loc[fund] = np.append(score, weight)
        count += 1
        print count,

    df_fund = base_ra_fund.load()
    df_fund = df_fund.loc[:, ['ra_code', 'ra_name']]
    df_fund = df_fund.set_index('ra_code')
    df_result = pd.merge(df_fund, df_result, left_index = True, right_index = True, how = 'right')
    df_result.to_csv('data/factor/fund_attrib.csv', index_label = 'date', encoding = 'gb2312')

    df_weight = pd.merge(df_fund, df_weight, left_index = True, right_index = True, how = 'right')
    df_weight.to_csv('data/factor/fund_factor_weight.csv', index_label = 'date', encoding = 'gb2312')

    set_trace()


def main():

    factor_nav = load_factor()
    # fund_nav = load_fund()
    fund_nav = load_fund_all()
    sdate = '2013-01-01'
    edate = '2018-01-01'

    # ortho_factor_recognize(factor_nav, sdate, edate)
    # ortho_factor_ret(factor_nav, sdate, edate)
    # fund_ret_attrib(factor_nav, fund_nav, sdate, edate)
    # factor_ortho(factor_nav, sdate, edate)
    ortho_fund_ret(fund_nav, sdate, edate)
    # ortho_fund_recognize(fund_nav, factor_nav, sdate, edate)


if __name__ == '__main__':

    # load_factor()
    # load_fund()
    # load_factor_nav()
    main()
