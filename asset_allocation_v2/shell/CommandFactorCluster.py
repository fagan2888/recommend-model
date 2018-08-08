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
from scipy.stats import rankdata, spearmanr, pearsonr
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import SpectralClustering
from scipy.spatial import distance_matrix
import statsmodels.api as sm
import datetime
from ipdb import set_trace
import warnings
warnings.filterwarnings('ignore')

import Portfolio as PF
import Const
from db import asset_ra_pool_nav, asset_ra_pool_fund, asset_ra_pool, base_ra_fund_nav, base_ra_fund, base_ra_index, asset_ra_composite_asset_nav, database
# from CommandMarkowitz import load_nav_series
from trade_date import ATradeDate
from asset import Asset

@click.group(invoke_without_command=True)
@click.option('--id', 'optid', help='specify markowitz id')
@click.pass_context
def fc(ctx, optid):
    '''
    factor layereing
    '''
    if ctx.invoked_subcommand is None:
        # ctx.invoke(fc_update, optid)
        ctx.invoke(fc_low, optid = optid)
        ctx.invoke(fc_high, optid = optid)
    else:
        pass


@fc.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fc_low_test(ctx, optid):

    lookback_days = 365 * 15
    factor_ids_1 = ['120000013', '120000020', '120000014', '120000015']
    factor_ids_2 = ['120000010', '120000011', '120000039']
    factor_ids_3 = ['120000053', '120000056', '120000058', '120000073', 'MZ.F00010', 'MZ.F00050', 'MZ.F00060', 'MZ.F00070', 'MZ.F10010',]
    factor_ids = factor_ids_1 + factor_ids_2 + factor_ids_3
    trade_dates = ATradeDate.month_trade_date(begin_date = '2018-01-01')
    for date in trade_dates:
        start_date = (date - datetime.timedelta(lookback_days)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')
        print(start_date, end_date)
        df_std_dist = load_mv(factor_ids, start_date, end_date)
        df_std_dist = df_std_dist / 10
        _, asset_cluster, _ = clusterKMeansLow(df_std_dist, n_init=10)
        asset_cluster = dict(list(zip(sorted(asset_cluster), sorted(asset_cluster.values()))))

        for k,v in asset_cluster.items():
            print(v)
        print()


@fc.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fc_low(ctx, optid):

    years = 15
    lookback_days = 365 * years
    factor_ids_1 = ['120000013', '120000014', '120000015', '120000028', '120000029']
    factor_ids_2 = ['120000010', '120000011', '120000039']
    factor_ids_3 = ['120000053', '120000056', '120000058', '120000073']
    factor_ids = factor_ids_1 + factor_ids_2 + factor_ids_3
    trade_dates = ATradeDate.month_trade_date(begin_date = '2017-01-01')
    for date in trade_dates:
        start_date = (date - datetime.timedelta(lookback_days)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')
        print(start_date, end_date)

        corr0 = load_corr(factor_ids, start_date, end_date)
        std0 = load_std(factor_ids, start_date, end_date)
        # asset_cluster = clusterSpectral(corr0*std0)
        # _, asset_cluster, _ = clusterKMeansHigh(corr0*std0, n_init=10)
        asset_cluster = clusterSimple(corr0, std0, years)
        asset_cluster = dict(list(zip(sorted(asset_cluster), sorted(asset_cluster.values()))))

        for k,v in asset_cluster.items():
            print(v)
        print()



@fc.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fc_high(ctx, optid):

    years = 5
    lookback_days = 365 * years
    factor_ids_1 = ['120000013', '120000015', '120000020', '120000014', '120000028']
    factor_ids_2 = ['120000016', '120000051', '120000056', '120000073', 'MZ.FA0010', 'MZ.FA0050', 'MZ.FA0070', 'MZ.FA1010']
    factor_ids = factor_ids_1 + factor_ids_2
    trade_dates = ATradeDate.month_trade_date(begin_date = '2017-01-01')
    for date in trade_dates:
        start_date = (date - datetime.timedelta(lookback_days)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')
        print(start_date, end_date)
        corr0 = load_corr(factor_ids, start_date, end_date)
        std0 = load_std(factor_ids, start_date, end_date)
        asset_cluster = clusterSimple(corr0, std0**3, years)
        asset_cluster = dict(list(zip(sorted(asset_cluster), sorted(asset_cluster.values()))))

        for k,v in asset_cluster.items():
            print(v)
        print()


def clusterKMeansLow(corr0,n_init=100):
    dist,silh=((1-corr0.fillna(0))/2.)**.5,pd.Series()
    # distance matrix
    for init in range(n_init):
        for i in range(3, 4):
    # find optimal num clusters
            kmeans_ = KMeans(n_clusters=i,n_jobs=1,n_init=1)
            kmeans_ = kmeans_.fit(dist)
            silh_= silhouette_samples(dist,kmeans_.labels_)
            stat = (silh_.mean()/silh_.std(),silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0]>stat[1]:
                silh,kmeans=silh_,kmeans_
            # print init, i, silh

    # n_clusters = len( np.unique( kmeans.labels_ ) )
    newIdx=np.argsort(kmeans.labels_)
    corr1=corr0.iloc[newIdx] # reorder rows
    corr1=corr1.iloc[:,newIdx] # reorder columns
    clstrs={i:corr0.columns[np.where(kmeans.labels_==i)[0] ].tolist() for i in np.unique(kmeans.labels_) } # cluster members
    silh=pd.Series(silh,index=dist.index)

    return corr1,clstrs,silh


def clusterKMeansHigh(corr0,n_init=100):
    dist,silh=((1-corr0.fillna(0))/2.)**.5,pd.Series()
    # distance matrix
    for init in range(n_init):
        for i in range(4, 5):
            # find optimal num clusters
            kmeans_ = KMeans(n_clusters=i,n_jobs=1,n_init=1)
            kmeans_ = kmeans_.fit(dist)
            silh_= silhouette_samples(dist,kmeans_.labels_)
            stat = (silh_.mean()/silh_.std(),silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0]>stat[1]:
                silh,kmeans=silh_,kmeans_

    newIdx=np.argsort(kmeans.labels_)
    corr1=corr0.iloc[newIdx]
    corr1=corr1.iloc[:,newIdx]
    clstrs={i:corr0.columns[np.where(kmeans.labels_==i)[0] ].tolist() for i in np.unique(kmeans.labels_) } # cluster members
    silh=pd.Series(silh,index=dist.index)

    return corr1,clstrs,silh


def clusterSpectral(feature, n_clusters = 4):

    feature =((1-feature.fillna(0))/2.)**.5
    mod = SpectralClustering(n_clusters = n_clusters, eigen_solver = 'arpack', random_state = 10, n_init = 100, n_jobs = 10)
    res = mod.fit(feature)
    cluster = {}
    assets = feature.columns.values
    for label in res.labels_:
        cluster[label] = list(assets[res.labels_ == label])

    return cluster


def clusterSimple(corr0, std0, years):

    threshold = 3 / years
    threshold = min([0.6, threshold])
    corr0 = corr0 * std0
    asset_cluster = {}
    factor_ids = corr0.keys()
    asset_cluster[0] = [factor_ids[0]]
    for factor_id in factor_ids[1:]:
        flag = False
        new_layer = len(asset_cluster)
        for layer in asset_cluster.keys():
            tmp_corr = corr0.loc[factor_id, asset_cluster[layer]].values.mean()
            if tmp_corr > threshold:
                flag = True
                asset_cluster[layer].append(factor_id)
        if not flag:
            asset_cluster[new_layer] = [factor_id]

    return asset_cluster


def load_corr(factor_ids, start_date, end_date):

    trade_dates = ATradeDate.trade_date(start_date, end_date)
    asset_navs = {}
    for factor_id in factor_ids:
        # asset_navs[factor_id] = CommandMarkowitz.load_nav_series(factor_id, reindex = trade_dates)
        asset_navs[factor_id] = Asset.load_nav_series(factor_id, reindex = trade_dates)

    df_asset_navs = pd.DataFrame(asset_navs)
    # df_asset_incs = df_asset_navs.pct_change().dropna()
    # corr = df_asset_incs.corr()
    corr = df_asset_navs.corr()

    return corr


def load_std(factor_ids, start_date, end_date):

    trade_dates = ATradeDate.trade_date(start_date, end_date)
    asset_navs = {}
    for factor_id in factor_ids:
        asset_navs[factor_id] = Asset.load_nav_series(factor_id, reindex = trade_dates)

    df_asset_navs = pd.DataFrame(asset_navs)
    df_asset_incs = df_asset_navs.pct_change().dropna()
    df_std = df_asset_incs.std()
    values = np.zeros((len(df_std), len(df_std)))
    for i in range(len(df_std)):
        for j in range(len(df_std)):
            tmp_value = df_std.iloc[i] / df_std.iloc[j]
            if tmp_value > 1:
                tmp_value = 1 / tmp_value
            values[i, j] = tmp_value

    df_dist = pd.DataFrame(data = values, columns = df_asset_navs.columns, index = df_asset_navs.columns)

    return df_dist


def load_ret(factor_ids, start_date, end_date):

    trade_dates = ATradeDate.trade_date(start_date, end_date)
    asset_navs = {}
    for factor_id in factor_ids:
        asset_navs[factor_id] = Asset.load_nav_series(factor_id, reindex = trade_dates)

    df_asset_navs = pd.DataFrame(asset_navs)
    df_asset_incs = df_asset_navs.pct_change().dropna()
    df_std = df_asset_incs.mean()
    values = np.zeros((len(df_std), len(df_std)))
    for i in range(len(df_std)):
        for j in range(len(df_std)):
            tmp_value = df_std.iloc[i] / df_std.iloc[j]
            if tmp_value > 1:
                tmp_value = 1 / tmp_value
            values[i, j] = tmp_value

    df_dist = pd.DataFrame(data = values, columns = df_asset_navs.columns, index = df_asset_navs.columns)

    return df_dist


def load_mv(factor_ids, start_date, end_date):

    trade_dates = ATradeDate.trade_date(start_date, end_date)
    asset_navs = {}
    for factor_id in factor_ids:
        asset_navs[factor_id] = Asset.load_nav_series(factor_id, reindex = trade_dates)

    df_asset_navs = pd.DataFrame(asset_navs)
    df_asset_incs = df_asset_navs.pct_change().dropna()
    df_mv = df_asset_incs.mean() / df_asset_incs.std()
    df_mv = df_mv.to_frame('mv')
    mv_dist = distance_matrix(df_mv, df_mv)
    df_dist = pd.DataFrame(data = mv_dist, columns = df_asset_navs.columns, index = df_asset_navs.columns)

    return df_dist






if  __name__ == '__main__':

    pass
