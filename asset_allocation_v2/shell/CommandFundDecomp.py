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
def fd_decomposition(ctx, optid):

    start_date = '2012-01-01'
    end_date = '2018-09-01'
    lookback = 360 * 4
    layer_num = 30

    # valid_funds = asset_fund.load_type_fund(l2codes=['200101', '200201']).index
    # df_nav_fund = pd.read_csv('data/df_nav_fund.csv', index_col=['date'], parse_dates=['date'])

    valid_funds = asset_fund.load_type_fund(l1codes=['2001', '2002']).index
    df_nav_fund = base_ra_fund_nav.load_daily(start_date, end_date, codes=valid_funds)
    df_nav_fund.to_csv('data/df_nav_fund_all.csv', index_label='date')
    # df_nav_fund = pd.read_csv('data/df_nav_fund_all.csv', index_col=['date'], parse_dates=['date'])

    fd_decomposition_days(df_nav_fund, lookback, layer_num)


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
        self.C = 0.3
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



