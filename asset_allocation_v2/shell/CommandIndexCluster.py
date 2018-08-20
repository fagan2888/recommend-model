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

from db import asset_ra_pool_nav, asset_ra_pool_fund, asset_ra_pool, base_ra_fund_nav, base_ra_fund, base_ra_index, asset_ra_composite_asset_nav, database, asset_stock_factor, asset_fund, asset_fund_factor, asset_index, asset_index_factor
# from CommandMarkowitz import load_nav_series
from trade_date import ATradeDate
from asset import Asset



@click.group(invoke_without_command=True)
@click.option('--id', 'optid', help='specify markowitz id')
@click.pass_context
def ic(ctx, optid):
    '''
    factor layereing
    '''
    if ctx.invoked_subcommand is None:
        ctx.invoke(ic_cluster, optid = optid)
    else:
        pass


@ic.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def ic_fund_style(ctx, optid):

    df_stability = pd.read_csv('data/factor/stability/fund_stability_mean.csv', index_col = ['date'], parse_dates = ['date'], encoding = 'gb2312')
    df_stability.index = ['%06d'%i for i in df_stability.index]

    valid_funds = df_stability[df_stability.stability > 0.0].index.values
    ffe = asset_fund_factor.load_fund_factor_exposure(fund_ids = valid_funds)

    fn = base_ra_fund.load(codes = valid_funds)
    fn = fn.set_index('ra_code')
    fn = fn.loc[:, ['ra_name']]

    all_indexes = asset_index.load_all_index_factor(if_type = 9)
    # all_indexes = asset_index.load_all_index_factor()
    valid_indexes = all_indexes.index.values
    ife = asset_index_factor.load_index_factor_exposure(index_ids = valid_indexes)

    dates = ife.index.levels[2]
    dates = dates[::-6][::-1]
    dates = dates[-2:]

    for ldate, date in zip(dates[:-1], dates[1:]):

        tffe = cal_feature(ffe, ldate, date)
        tife = cal_index_feature(ife, date)
        # df_style, index_pool = cal_fund_style(tffe, tife, 0.90)
        df_style, index_pool = cal_fund_style2(tffe, tife, 1.00)

        for index_id in sorted(index_pool.keys()):
            index_funds = index_pool.get(index_id)
            if len(index_funds) > 0:
                print()
                print(all_indexes.loc[index_id, 'index_name'])
                print(index_id)
                print(fn.loc[index_funds])

        set_trace()


@ic.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def ic_cluster(ctx, optid):


    # all_indexes = asset_index.load_all_index_factor(if_type = 9)
    all_indexes = asset_index.load_all_index_factor()
    valid_indexes = all_indexes.index.values
    ife = asset_index_factor.load_index_factor_exposure(index_ids = valid_indexes)

    dates = ife.index.levels[2]
    dates = dates[::-6][::-1]
    dates = dates[-2:-1]

    for date in dates:

        tife = cal_index_feature(ife, date)
        df_dist = tife.T.corr()
        asset_cluster = clusterSimple(df_dist, 0.9)
        clusters = sorted(asset_cluster, key = lambda x: len(asset_cluster[x]), reverse = True)
        for layer in clusters:
            print(layer)
            indexes = asset_cluster[layer]
            print(all_indexes.loc[indexes])


@ic.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def ic_cluster2(ctx, optid):


    # all_indexes = asset_index.load_all_index_factor(if_type = 9)
    all_indexes = asset_index.load_all_index_factor()
    valid_indexes = all_indexes.index.values
    ife = asset_index_factor.load_index_factor_exposure(index_ids = valid_indexes)

    dates = ife.index.levels[2]
    dates = dates[::-6][::-1]
    dates = dates[-2:-1]

    for date in dates:

        tife = cal_index_feature(ife, date)

        dist = distance_matrix(tife, tife)
        df_dist = pd.DataFrame(data = dist, index = tife.index, columns = tife.index)

        asset_cluster = clusterSimple2(df_dist, 1.0)
        clusters = sorted(asset_cluster, key = lambda x: len(asset_cluster[x]), reverse = True)
        for layer in clusters:
            print(layer)
            indexes = asset_cluster[layer]
            print(all_indexes.loc[indexes])


def cal_feature(ffe, sdate, edate):

    ffe = ffe.reset_index()
    ffe = ffe[(ffe.trade_date >= sdate) & (ffe.trade_date <= edate)]
    ffe = ffe[['fund_id', 'ff_id', 'exposure']].groupby(['fund_id', 'ff_id']).last().unstack()
    ffe.columns = ffe.columns.levels[1]
    ffe = ffe.dropna()

    return ffe


def cal_index_feature(ife, date):

    ife = ife[ife.index.get_level_values(2) == date]
    ife = ife.reset_index()
    ife = ife.set_index(['index_id', 'if_id'])
    ife = ife[['exposure']]
    ife = ife.unstack()
    ife.columns = ife.columns.levels[1]

    return ife


def cal_fund_style(tffe, tife, threshold):

    index_ids = tife.index
    fund_ids = tffe.index
    tife.columns = tffe.columns
    df_style = pd.concat([tffe, tife])
    df_style = df_style.T.corr()
    df_style = df_style.loc[fund_ids, index_ids]

    index_pool = {}
    for index_id in index_ids:
        index_pool[index_id] = []

    for fund_id in fund_ids:
        fund_style = df_style.loc[fund_id]
        max_corr = fund_style.max()
        max_corr_index = fund_style.argmax()

        if max_corr > threshold:
            index_pool[max_corr_index].append(fund_id)


    return df_style, index_pool


def cal_fund_style2(tffe, tife, threshold):

    index_ids = tife.index
    fund_ids = tffe.index
    tife.columns = tffe.columns
    df_style = pd.concat([tffe, tife])

    dist = distance_matrix(df_style, df_style)
    df_style = pd.DataFrame(data = dist, index = df_style.index, columns = df_style.index)
    df_style = df_style.loc[fund_ids, index_ids]

    index_pool = {}
    for index_id in index_ids:
        index_pool[index_id] = []

    for fund_id in fund_ids:
        fund_style = df_style.loc[fund_id]
        min_dist = fund_style.min()
        min_dist_index = fund_style.argmin()

        if min_dist < threshold:
            index_pool[min_dist_index].append(fund_id)


    return df_style, index_pool


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
            # tmp_corrs[layer] = dist.loc[factor_id, asset_cluster[layer]].values.min()
            tmp_corrs[layer] = dist.loc[factor_id, asset_cluster[layer]].values.mean()
            tmp_corrs_ser = pd.Series(tmp_corrs)
            tmp_corrs_ser = tmp_corrs_ser.sort_values(ascending = False)
        if (tmp_corrs_ser.iloc[0] > threshold) and (not flag):
            flag = True
            asset_cluster[tmp_corrs_ser.index[0]].append(factor_id)
        if not flag:
            asset_cluster[new_layer] = [factor_id]

    return asset_cluster


def clusterSimple2(dist, threshold):

    asset_cluster = {}
    factor_ids = dist.keys()
    asset_cluster[0] = [factor_ids[0]]
    for factor_id in factor_ids[1:]:
        print(factor_id)
        flag = False
        new_layer = len(asset_cluster)
        tmp_corrs = {}
        for layer in asset_cluster.keys():
            tmp_corrs[layer] = dist.loc[factor_id, asset_cluster[layer]].values.max()
            tmp_corrs_ser = pd.Series(tmp_corrs)
            tmp_corrs_ser = tmp_corrs_ser.sort_values(ascending = True)
        if (tmp_corrs_ser.iloc[0] < threshold) and (not flag):
            flag = True
            asset_cluster[tmp_corrs_ser.index[0]].append(factor_id)
        if not flag:
            asset_cluster[new_layer] = [factor_id]

    return asset_cluster


def load_market_indexes():

    stock_funds = asset_fund.load_fund_by_type(l1codes = 2001)
    set_trace()





if __name__ == '__main__':

    # fund_stability_filter()
    # fund_concentration_filter()
    load_market_indexes()










