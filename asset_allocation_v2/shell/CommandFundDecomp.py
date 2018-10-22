# coding=utf-8


import pandas as pd
import numpy as np
from sqlalchemy import MetaData, Table, select, func, literal_column
import matplotlib
import matplotlib.pyplot as plt
import click
import sys
sys.path.append('shell/')
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata, spearmanr, pearsonr, linregress
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, AffinityPropagation
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
import statsmodels.api as sm
import datetime
from collections import defaultdict
from ipdb import set_trace
import warnings
warnings.filterwarnings('ignore')

from db import asset_ra_pool_nav, asset_ra_pool_fund, asset_ra_pool, base_ra_fund_nav, base_ra_fund, base_ra_index, asset_ra_composite_asset_nav, database, asset_stock_factor, asset_fund_factor, asset_fund, asset_index, asset_index_factor
# from CommandMarkowitz import load_nav_series
from trade_date import ATradeDate
from asset import Asset

@click.group(invoke_without_command=True)
@click.option('--id', 'optid', help='specify fd id')
@click.pass_context
def fd(ctx, optid):
    '''
    factor decomposition
    '''
    if ctx.invoked_subcommand is None:
        ctx.invoke(fd_decomposition, optid=optid)
    else:
        pass


@fd.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fd_index_recognize(ctx, optid):

    layers = [
        ['240017', '270008', '373020', '377240', '660001', '660004', '660005'],
        ['020015', '020023', '110012', '110015', '519021'],
        ['150103', '151001', '310358', '470009', '519668'],
        ['110001', '110013', '270028', '470028'],
    ]

    df_nav_fund = pd.read_csv('data/df_nav_fund_all.csv', index_col=['date'], parse_dates=['date'])
    df_ret_fund = df_nav_fund.dropna(1).pct_change().dropna()
    df_nav_fund = df_nav_fund / df_nav_fund.iloc[0]

    estclass = ['中证策略指数', '上证策略指数', '国证策略指数', '深证策略指数', '申万量化策略指数']
    # valid_index_info = asset_index.load_type_index(estclass)
    valid_index_info = asset_index.load_type_index(estclasstype='中证%')
    valid_index = valid_index_info.index
    df_nav_index = asset_index.load_caihui_index(valid_index)
    df_nav_index = df_nav_index.reindex(df_nav_fund.index)
    df_ret_index = df_nav_index.dropna(1).pct_change().dropna()
    for layer in layers:
        tmp_ret = pd.concat([df_ret_index, df_ret_fund[layer]], 1)
        tmp_corr = tmp_ret.corr()
        tmp_corr = tmp_corr.loc[df_ret_index.columns, layer]
        tmp_corr = tmp_corr.mean(1)
        tmp_corr = tmp_corr.sort_values()
        print(layer)
        print(tmp_corr.tail())
    tn = df_nav_index.loc[:, ['2070012293', '2070007226', '2070006896', '2070007225', '2070012295', '2070012943']]
    set_trace()


@fd.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fd_index_cluster(ctx, optid):

    estclass = ['中证基金指数']
    valid_index_info = asset_index.load_type_index(estclass)
    valid_index = valid_index_info.index
    df_nav_index = asset_index.load_caihui_index(valid_index)
    df_nav_index = df_nav_index.loc['2012':].fillna(method='pad').dropna(1)
    df_nav_index = df_nav_index / df_nav_index.iloc[0]
    df_ret_index = df_nav_index.pct_change().dropna()

    df_dist = df_ret_index.corr()
    asset_cluster = cluster_ap(df_dist, 0.9)
    # asset_cluster = clusterSimple(df_dist, 0.9)
    clusters = sorted(asset_cluster, key=lambda x: len(asset_cluster[x]), reverse=True)
    for cluster in clusters:
        tmp_indexes = asset_cluster[cluster]
        if len(tmp_indexes) > 1:
            print(tmp_indexes)
    set_trace()


@fd.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fd_cluster(ctx, optid):

    start_date = '2012-01-01'
    end_date = '2018-09-01'
    lookback = 360 * 4
    layer_num = 10

    df_nav_fund = pd.read_csv('data/df_nav_fund_all.csv', index_col=['date'], parse_dates=['date'])
    df_valid_fund = pd.read_csv('data/fund_decomp/valid_fund.csv', index_col=['fund_id'])
    valid_fund_ids = df_valid_fund.index
    valid_fund_ids = ['%06d' % i for i in valid_fund_ids]
    # df_nav_fund = df_nav_fund.loc[:, valid_fund_ids]

    df_ret_fund = df_nav_fund.dropna(1).pct_change().dropna()
    # df_nav_fund = df_nav_fund / df_nav_fund.iloc[0]
    df_dist = df_ret_fund.corr()
    df_dist = cal_dcor_dist(df_ret_fund)
    asset_cluster = cluster_ap(df_dist, 0.9)
    # asset_cluster = clusterSimple(df_dist, 0.9)
    clusters = sorted(asset_cluster, key=lambda x: len(asset_cluster[x]), reverse=True)
    for cluster in clusters:
        tmp_funds = asset_cluster[cluster]
        if len(tmp_funds) > 1:
            print(tmp_funds)
    set_trace()


@fd.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def index_cluster(ctx, optid):

    index_ids = ['120000001', '120000002', '120000013', '120000015', '120000014', '120000020']
    df_nav_index = {}
    for index_id in index_ids:
        tmp_nav = Asset.load_nav_series(index_id)
        df_nav_index[index_id] = tmp_nav
    df_nav_index = pd.DataFrame(df_nav_index)
    df_ret_index = df_nav_index.pct_change().dropna()
    df_ret_index = df_ret_index.replace(0.0, np.nan).dropna()
    df_ret_index = df_ret_index.loc['2015']
    # df_dist = df_ret_fund.corr()
    df_dist = cal_dcor_dist(df_ret_index)
    asset_cluster = cluster_ap(df_dist, 0.9)
    # asset_cluster = clusterSimple(df_dist, 0.9)
    clusters = sorted(asset_cluster, key=lambda x: len(asset_cluster[x]), reverse=True)
    for cluster in clusters:
        tmp_funds = asset_cluster[cluster]
        if len(tmp_funds) > 1:
            print(tmp_funds)
    set_trace()


@fd.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fd_decomposition(ctx, optid):

    start_date = '2012-01-01'
    end_date = '2018-09-01'
    lookback = 360 * 4
    layer_num = 10

    # valid_funds = asset_fund.load_type_fund(l2codes=['200101', '200201']).index
    # df_nav_fund = pd.read_csv('data/df_nav_fund.csv', index_col=['date'], parse_dates=['date'])

    # valid_funds = asset_fund.load_type_fund(l1codes=['2001', '2002']).index
    # df_nav_fund = base_ra_fund_nav.load_daily(start_date, end_date, codes=valid_funds)
    # df_nav_fund.to_csv('data/df_nav_fund_all.csv', index_label='date')

    df_nav_fund = pd.read_csv('data/df_nav_fund_all.csv', index_col=['date'], parse_dates=['date'])
    df_valid_fund = pd.read_csv('data/fund_decomp/valid_fund.csv', index_col=['fund_id'])
    valid_fund_ids = df_valid_fund.index
    valid_fund_ids = ['%06d' % i for i in valid_fund_ids]
    df_nav_fund = df_nav_fund.loc[:, valid_fund_ids]

    fd_decomposition_days(df_nav_fund, lookback, layer_num)


@fd.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fd_decomposition(ctx, optid):

    start_date = '2012-01-01'
    end_date = '2018-09-01'
    lookback = 360 * 4
    layer_num = 10

    # valid_funds = asset_fund.load_type_fund(l2codes=['200101', '200201']).index
    # df_nav_fund = pd.read_csv('data/df_nav_fund.csv', index_col=['date'], parse_dates=['date'])

    # valid_funds = asset_fund.load_type_fund(l1codes=['2001', '2002']).index
    # df_nav_fund = base_ra_fund_nav.load_daily(start_date, end_date, codes=valid_funds)
    # df_nav_fund.to_csv('data/df_nav_fund_all.csv', index_label='date')

    df_nav_fund = pd.read_csv('data/df_nav_fund_all.csv', index_col=['date'], parse_dates=['date'])
    df_valid_fund = pd.read_csv('data/fund_decomp/valid_fund.csv', index_col=['fund_id'])
    valid_fund_ids = df_valid_fund.index
    valid_fund_ids = ['%06d' % i for i in valid_fund_ids]
    df_nav_fund = df_nav_fund.loc[:, valid_fund_ids]

    fd_decomposition_days(df_nav_fund, lookback, layer_num)


@fd.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fd_identification(ctx, optid):

    df_nav_decomp = pd.read_csv('data/fund_decomp/fund_decomp_nav.csv', index_col=['date'], parse_dates=['date'])
    df_weight_decomp = pd.read_csv('data/fund_decomp/fund_decomp_weight.csv', index_col=['date'], parse_dates=['date'])
    ind_ids = ['1200000%2d' % i for i in range(52, 80)]
    style_ids = ['MZ.FA00%d0' % i for i in range(1, 10)] + ['MZ.FA10%d0' % i for i in range(1, 10)]
    index_ids = ind_ids + style_ids
    df_nav_index = {}
    for index_id in index_ids:
        tmp_nav = Asset.load_nav_series(index_id)
        df_nav_index[index_id] = tmp_nav
    df_nav_index = pd.DataFrame(df_nav_index)

    new_factors = []
    df_ret_decomp = df_nav_decomp.pct_change().dropna()
    df_nav_index = df_nav_index.reindex(df_nav_decomp.index)
    df_ret_index = df_nav_index.pct_change().dropna()
    for layer in df_nav_decomp.columns:
        df_ret_joint = pd.concat([df_ret_index, df_ret_decomp[layer]], 1)
        df_corr = df_ret_joint.corr().loc[layer].sort_values()
        max_corr = df_corr.iloc[-2]
        if max_corr < 0.90:
            new_factors.append(layer)
        print(layer, df_corr.index[-2], df_corr.iloc[-2])

    df_factor_nav = df_nav_decomp[new_factors]
    df_factor_weight = df_weight_decomp[new_factors]
    for factor in new_factors:
        tmp_weight = df_factor_weight[factor].sort_values()
        tmp_nav = df_factor_nav[factor]
        print(tmp_nav.iloc[-1])
        print(tmp_weight.tail(5))


@fd.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fd_search(ctx, optid):

    df_nav_fund = pd.read_csv('data/df_nav_fund_all.csv', index_col=['date'], parse_dates=['date'])
    ind_ids = ['1200000%2d' % i for i in range(52, 80)] + ['120000001', '120000002']
    style_ids = ['MZ.FA00%d0' % i for i in range(1, 10)] + ['MZ.FA10%d0' % i for i in range(1, 10)]
    index_ids = ind_ids + style_ids
    df_nav_index = {}
    for index_id in index_ids:
        tmp_nav = Asset.load_nav_series(index_id)
        df_nav_index[index_id] = tmp_nav
    df_nav_index = pd.DataFrame(df_nav_index)

    df_ret_fund = df_nav_fund.dropna(1).pct_change().dropna()
    df_nav_fund = df_nav_fund.dropna(1)
    df_nav_fund = df_nav_fund / df_nav_fund.iloc[0]
    df_nav_index = df_nav_index.reindex(df_nav_fund.index)
    df_ret_index = df_nav_index.dropna(1).pct_change().dropna()

    df_res = pd.DataFrame(columns=['corr', 'nav'])
    for fund in df_ret_fund.columns:
        tmp_ret = pd.concat([df_ret_index, df_ret_fund[[fund]]], 1)
        tmp_corr = tmp_ret.corr().loc[fund].iloc[:-1].max()
        final_nav = df_nav_fund[fund].iloc[-1]
        df_res.loc[fund] = [tmp_corr, final_nav]
        print(fund, tmp_corr, final_nav)
    df_res = df_res[df_res['nav'] > 1.5]
    df_res = df_res[df_res['corr'] < 0.9]

    df_res.to_csv('data/fund_decomp/valid_fund.csv', index_label=['fund_id'])


def fd_decomposition_days(df_nav_fund, lookback, layer_num):

    dates = df_nav_fund.index
    # for sdate, edate in zip(dates[:-lookback], dates[lookback:]):
    for sdate, edate in [(dates[0], dates[-1])]:
        df_nav_fund = df_nav_fund.loc[sdate:edate, :]
        fd_decomposition_day(df_nav_fund, layer_num)


def fd_decomposition_day(df_nav_fund, layer_num):

    dfr = df_nav_fund.dropna(1).pct_change().dropna()
    # constraint_pca_iter(dfr, layer_num)
    cpca = ConstraintPca(dfr, layer_num)
    cpca.handle()

    # pca = PCA(n_components=layer_num)
    # pca.fit(dfr)
    # constraint_pca(dfr, layer_num)


class ConstraintPca(object):

    def __init__(self, dfr, layer_num):

        self.dfr = dfr
        self.layer_num = layer_num
        self.C = 0.0
        self.weights = []

    def constraint_pca(self, dfr):

        fund_num = dfr.shape[1]
        w0 = np.zeros(fund_num)
        w0[0] = 1.0
        V = dfr.cov()

        cons = (
            {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        )

        res = minimize(self.pca_objective, w0, args=[V], method='SLSQP', constraints=cons, options={'disp': False})
        self.weights.append(res.x / res.x.sum())

        return res.x

    def constraint_pca_iter(self, dfr):

        fund_num = dfr.shape[1]
        w0 = np.ones(fund_num) / fund_num

        cons = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x},
        )

        res = minimize(self.corr_objective, w0, args=[dfr, self.weights], method='SLSQP', constraints=cons, options={'disp': False})
        print('max corr:', res.fun - res.x.var()*len(res.x)*self.C)
        weights = res.x / res.x.sum()
        print(np.sort(weights)[-5:])
        self.weights.append(weights)

        return res.x

    def pca_objective(self, x, pars):

        V = pars[0]
        S = np.dot(x, np.dot(x, V))
        loss = -S

        return loss

    def corr_objective(self, x, pars):

        dfr = pars[0]
        weights = pars[1]
        S = []
        for weight in weights:
            tmp_corr = np.corrcoef(np.dot(dfr, x), np.dot(dfr, weight))[1, 0]
            S.append(tmp_corr)
        # S = np.mean(S)
        S = np.max(S)
        S += (x.var()*len(x)) * self.C
        # S = np.corrcoef(np.dot(dfr, x), np.dot(dfr, last_w))[1, 0]

        return S

    def handle(self):

        self.constraint_pca(self.dfr)
        for i in range(1, self.layer_num):
            print("calculating layer %d ..." % (i + 1))
            self.constraint_pca_iter(self.dfr)

        data = {}
        for i in range(self.layer_num):
            data['layer%d' % i] = np.dot(self.dfr, self.weights[i])
        df = pd.DataFrame(data=data, index=self.dfr.index)
        df = (1 + df).cumprod()
        # df.to_csv('data/fund_decomp/fund_decomp_nav.csv', index_label='date')

        weight = {}
        for i in range(self.layer_num):
            weight['layer%d' % i] = self.weights[i]
        df_weight = pd.DataFrame(data=weight, index=self.dfr.columns)
        # df_weight.to_csv('data/fund_decomp/fund_decomp_weight.csv', index_label='date')
        set_trace()


def constraint_pca(dfr, layer_num):

    fund_num = dfr.shape[1]
    # w0 = np.ones(fund_num) / fund_num
    w0 = np.zeros(fund_num)
    w0[0] = 1.0
    V = dfr.cov()

    cons = (
        # {'type': 'eq', 'fun': lambda x:((x.reshape(-1, layer_num).sum(0) - np.ones(layer_num))**2).sum()},
        # {'type': 'eq', 'fun': lambda x: x.sum(0) - 1},
        {'type': 'eq', 'fun': lambda x: np.dot(x, x) - 1},
        {'type': 'ineq', 'fun': lambda x: x}
    )

    res = minimize(pca_objective, w0, args=[V], method='SLSQP', constraints=cons, options={'disp': False})
    # res = minimize(pca_objective, w0, args=[dfr], method='SLSQP', options={'disp': False})

    return res.x


def constraint_pca_iter(dfr, layer_num):

    last_w = constraint_pca(dfr, layer_num)
    last_w / last_w.sum()
    fund_num = dfr.shape[1]
    w0 = np.ones(fund_num) / fund_num

    cons = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: x},
   )

    res = minimize(corr_objective, w0, args=[dfr, last_w], method='SLSQP', constraints=cons, options={'disp': False})
    # res = minimize(pca_objective, w0, args=[dfr], method='SLSQP', options={'disp': False})
    df = pd.DataFrame(
        data={
            'layer1': np.dot(dfr, last_w / last_w.sum()),
            'layer2': np.dot(dfr, res.x / res.x.sum()),
        },
        index=dfr.index
    )
    df = (1 + df).cumprod()

    return res.x


def corr_objective(x, pars):

    dfr = pars[0]
    last_w = pars[1]
    S = np.corrcoef(np.dot(dfr, x), np.dot(dfr, last_w))[1, 0]
    print(S)

    return S


def pca_objective(x, pars):

    V = pars[0]
    S = np.dot(x, np.dot(x, V))
    print(S)
    loss = -S

    return loss


def pca_objective_orig(x, pars):

    # x_mean = x.mean(1).reshape(-1, 1)
    dfr = pars[0]
    layer_num = pars[1]
    x = x.reshape(-1, layer_num)
    fr = np.dot(dfr, x)

    # rsquares = []
    # for i in range(dfr.shape[1]):
        # y = dfr.iloc[:, i]
        # res = sm.OLS(y, fr).fit()
        # rsquares.append(res.rsquared)
    # mean_rsquare = -np.mean(rsquares)
    p_var = np.cov(fr.T)
    diag = np.eye(layer_num)
    loss = -(diag * p_var).sum()
    print(loss)

    return loss


def cluster_ap(df_dist, preference=None):

    if preference is not None:
        ap_model = AffinityPropagation(affinity='precomputed', preference=preference)
    else:
        ap_model = AffinityPropagation(affinity='precomputed')

    try:
        ap_res = ap_model.fit(df_dist)
    except Exception as e:
        print(e)
        set_trace()

    # silh = silhouette_score(-df_dist, ap_model.labels_)
    asset_cluster = defaultdict(list)
    for label, fund_id in zip(ap_res.labels_, df_dist.index):
        asset_cluster[label].append(fund_id)

    return asset_cluster


def clusterSimple(dist, threshold):

    asset_cluster = {}
    factor_ids = dist.keys()
    asset_cluster[0] = [factor_ids[0]]
    for factor_id in factor_ids[1:]:
        # print(factor_id)
        flag = False
        new_layer = len(asset_cluster)
        tmp_corrs = {}
        for layer in asset_cluster.keys():
            tmp_corrs[layer] = dist.loc[factor_id, asset_cluster[layer]].values.mean()
            tmp_corrs_ser = pd.Series(tmp_corrs)
            max_corr_index = tmp_corrs_ser.argmax()
        if (tmp_corrs_ser.loc[max_corr_index] > threshold) and (not flag):
            flag = True
            asset_cluster[max_corr_index].append(factor_id)
        if not flag:
            asset_cluster[new_layer] = [factor_id]

    return asset_cluster


def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


def cal_dcor_dist(df_ret_fund):
    fund_ids = df_ret_fund.columns
    fund_num = len(fund_ids)
    data = np.zeros((fund_num, fund_num))
    for i in range(fund_num):
        for j in range(fund_num):
            tmp_dcor = distcorr(df_ret_fund.iloc[:, i].values, df_ret_fund.iloc[:, j].values)
            data[i, j] = tmp_dcor
    df_dist = pd.DataFrame(data=data, columns=fund_ids, index=fund_ids)
    return df_dist


def cal_mic_dist(df_ret_fund):
    set_trace()



