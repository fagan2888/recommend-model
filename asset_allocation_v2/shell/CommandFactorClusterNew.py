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
import networkx as nx
from networkx.algorithms import minimum_spanning_tree
import warnings
warnings.filterwarnings('ignore')

import Portfolio as PF
import Const
from db import asset_ra_pool_nav, asset_ra_pool_fund, asset_ra_pool, base_ra_fund_nav, base_ra_fund, base_ra_index, asset_ra_composite_asset_nav, database
import DBData
# from CommandMarkowitz import load_nav_series
import CommandMarkowitz
from trade_date import ATradeDate
from asset import Asset
from graphviz import PlotGraph

@click.group(invoke_without_command=True)
@click.option('--id', 'optid', help='specify markowitz id')
@click.pass_context
def fc(ctx, optid):
    '''
    factor layereing
    '''
    if ctx.invoked_subcommand is None:
        # ctx.invoke(fc_update, optid)
        ctx.invoke(fc_rolling, optid = optid)
        ctx.invoke(fc_update_nav, optid = optid)
    else:
        pass


@fc.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fc_rolling(ctx, optid):

    lookback_days = 365 * 15
    # blacklist = [24, 32, 40]
    # factor_ids = ['1200000%02d'%i for i in range(1, 40) if i not in blacklist]
    factor_ids_1 = ['120000013', '120000020', '120000014', '120000015']
    # factor_ids_1 = ['120000010', '120000011', '120000039', '120000013', '120000020', '120000014', '120000044', '120000015', '120000019', '120000034']
    # factor_ids_2 = ['1200000%d'%i for i in range(52, 80)]
    factor_ids_2 = ['120000010', '120000011', '120000039']
    # factor_ids_2 = ['120000052', '120000056', '120000058', '120000073', '120000078', '120000079']
    # factor_ids_3 = ['MZ.F000%d0'%i for i in range(1, 10)] + ['MZ.F100%d0'%i for i in range(1, 10)]
    # factor_ids_3 = ['120000053', '120000056', '120000058', '120000073', 'MZ.F00010', 'MZ.F00050', 'MZ.F00060', 'MZ.F00070', 'MZ.F10010',]
    factor_ids_3 = ['120000053', '120000056', '120000058', '120000073']
    # factor_ids = factor_ids_1 + factor_ids_2 + factor_ids_3
    factor_ids = factor_ids_1 + factor_ids_2 + factor_ids_3
    # factor_ids = factor_ids_3
    trade_dates = ATradeDate.month_trade_date(begin_date = '2018-01-01')
    for date in trade_dates:
        start_date = (date - datetime.timedelta(lookback_days)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')
        print(start_date, end_date)
        corr0 = load_ind(factor_ids, start_date, end_date)
        # corr0 = load_feature(factor_ids, start_date, end_date)
        # corr0 = corr0 / 10
        # factor_name = base_ra_index.load()
        set_trace()
        _, asset_cluster, _ = clusterKMeansBase(corr0, minNumCluster = 6, maxNumClusters=6, n_init=10)
        # asset_cluster = clusterSpectral(corr0)
        # asset_cluster = clusterSimple(corr0, threshold = 0.90)
        asset_cluster = dict(list(zip(sorted(asset_cluster), sorted(asset_cluster.values()))))

        for k,v in asset_cluster.items():
            # v = np.array(v).astype('int')
            # print(factor_name.loc[v])
            print(v)
        print()


@fc.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fc_mst(ctx, optid):

    # out_file = 'graph/mst_1'
    # blacklist = [56]
    lookback_days = 365
    # factor_ids_1 = ['120000042', '120000043', '120000044']
    # factor_ids_1 = ['120000042', '120000043']
    # factor_ids_1 = ['120000043']
    factor_ids_1 = []
    factor_ids_2 = ['1200000%d'%i for i in range(52, 80)]
    # factor_ids_2 = ['1200000%d'%i for i in range(52, 80) if i not in blacklist]
    # factor_ids_2 = ['120000052', '120000056', '120000058', '120000073', '120000078', '120000079']
    factor_ids_3 = ['MZ.F000%d0'%i for i in range(1, 10)] + ['MZ.F100%d0'%i for i in range(1, 10)]
    # factor_ids = factor_ids_1 + factor_ids_2 + factor_ids_3
    # factor_ids = factor_ids_1 + factor_ids_3
    factor_ids = factor_ids_2
    trade_dates = ATradeDate.month_trade_date(begin_date = '2017-01-01')
    for date in trade_dates:
        start_date = (date - datetime.timedelta(lookback_days)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')
        corr0 = load_ind(factor_ids, start_date, end_date)
        # G = nx.from_numpy_array(corr0)
        G = nx.from_pandas_adjacency(corr0)
        mst = minimum_spanning_tree(G)
        # df_res = pd.DataFrame(data = mod.toarray(), index = corr0.index, columns = corr0.columns)
        # PlotGraph.plot_graph(G, filename=out_file, colored_edges=mst.edges())
        PlotGraph.plot_graph(G, colored_edges=mst.edges())
        set_trace()


@fc.command()
@click.option('--id', 'optid', help='specify cluster id')
@click.pass_context
def fc_update_nav(ctx, optid):

    lookback_days = 365
    blacklist = [24, 32, 40]
    factor_ids = ['1200000%02d'%i for i in range(1, 40) if i not in blacklist]
    trade_dates = ATradeDate.month_trade_date(begin_date = '2018-01-01')
    date = trade_dates[-1]

    start_date = (date - datetime.timedelta(lookback_days)).strftime('%Y-%m-%d')
    end_date = date.strftime('%Y-%m-%d')
    corr0 = load_ind(factor_ids, start_date, end_date)
    res = clusterKMeansBase(corr0, maxNumClusters=10, n_init=100)
    asset_cluster = res[1]
    asset_cluster = dict(list(zip(sorted(asset_cluster), sorted(asset_cluster.values()))))

    factor_name = base_ra_index.load()
    for k,v in asset_cluster.items():
        v = np.array(v).astype('int')
        print(factor_name.loc[v])

    assets = {}
    for factor_id in factor_ids:
        assets[factor_id] = Asset.load_nav_series(factor_id)
    df_assets = pd.DataFrame(assets)

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('ra_composite_asset_nav', metadata, autoload=True)

    for layer in list(asset_cluster.keys()):
        layer_id = 'FC.000001.%d'%(layer+1)
        layer_assets = asset_cluster[layer]
        layer_nav = df_assets.loc[:,layer_assets]
        layer_ret = layer_nav.pct_change().dropna()
        layer_ret = layer_ret.mean(1)
        layer_ret = layer_ret.reset_index()
        layer_ret.columns = ['ra_date', 'ra_inc']
        layer_ret['ra_nav'] = (1 + layer_ret['ra_inc']).cumprod()
        layer_ret['ra_asset_id'] = layer_id
        df_new = layer_ret.set_index(['ra_asset_id', 'ra_date'])
        df_old = asset_ra_composite_asset_nav.load_nav(layer_id)
        df_new = df_new.reindex(columns = ['ra_nav', 'ra_inc'])
        database.batch(db, t, df_new, df_old, timestamp=False)


#def load_fund(start_date, end_date):
#    df_nav_fund = pd.read_csv('data/df_nav_fund.csv', index_col = ['td_date'], parse_dates = ['td_date'])
#    df_nav_inc = df_nav_fund.pct_change()
#    df_nav_inc = df_nav_inc.loc[start_date:end_date]
#    df_nav_inc = df_nav_inc.dropna(1)
#    # df_nav_inc = df_nav_inc.iloc[:, :28]
#    corr = df_nav_inc.corr()
#    return corr


def load_ind(factor_ids, start_date, end_date):

    # trade_dates = DBData.trade_dates(start_date, end_date)
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


def load_feature(factor_ids, start_date, end_date):

    # trade_dates = DBData.trade_dates(start_date, end_date)
    trade_dates = ATradeDate.trade_date(start_date, end_date)
    asset_navs = {}
    for factor_id in factor_ids:
        asset_navs[factor_id] = Asset.load_nav_series(factor_id, reindex = trade_dates)

    df_asset_navs = pd.DataFrame(asset_navs)
    df_asset_incs = df_asset_navs.pct_change(5).dropna()

    # df_feature = pd.DataFrame(columns = df_asset_navs.columns)
    # df_feature.loc['std'] = df_asset_incs.std()
    # df_feature.loc['corr'] = df_asset_incs.corr().mean()
    # df_feature.loc['ret'] = df_asset_incs.mean()
    df_feature = df_asset_incs.corr()

    dist = distance_matrix(df_feature.T, df_feature.T)
    df_dist = pd.DataFrame(data = dist, columns = df_asset_navs.columns, index = df_asset_navs.columns)

    return df_dist


def clusterKMeansBase(corr0,minNumCluster=5,maxNumClusters=10,n_init=100):
    dist,silh=((1-corr0.fillna(0))/2.)**.5,pd.Series()
    # distance matrix
    for init in range(n_init):
        for i in range(minNumCluster,maxNumClusters+1):
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


def makeNewOutputs(corr0, clstrs, clstrs2):
    clstrsNew,newIdx={},[]
    for i in list(clstrs.keys()):
        clstrsNew[len(list(clstrsNew.keys()))]=list(clstrs[i])
    for i in list(clstrs2.keys()):
        clstrsNew[len(list(clstrsNew.keys()))]=list(clstrs2[i])
    list(map(newIdx.extend, list(clstrsNew.values())))
    corrNew=corr0.loc[newIdx,newIdx]
    dist=((1-corr0.fillna(0))/2.)**.5
    kmeans_labels=np.zeros(len(dist.columns))
    for i in list(clstrsNew.keys()):
        idxs=[dist.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs]=i
    silhNew=pd.Series(silhouette_samples(dist,kmeans_labels),index=dist.index)
    return corrNew,clstrsNew,silhNew


def clusterKMeansTop(corr0, maxNumClusters=10, n_init=10):
    corr1,clstrs,silh=clusterKMeansBase(corr0,maxNumClusters=corr0.shape[1]-1,n_init=n_init)
    clusterTstats={i:np.mean(silh[clstrs[i]])/np.std(silh[clstrs[i]]) for i in list(clstrs.keys())}
    tStatMean=np.mean(list(clusterTstats.values()))
    redoClusters=[i for i in list(clusterTstats.keys()) if clusterTstats[i]<tStatMean]
    if len(redoClusters)<=2:
        return corr1,clstrs,silh
    else:
        keysRedo=[]
        list(map(keysRedo.extend,[clstrs[i] for i in redoClusters]))
        corrTmp=corr0.loc[keysRedo,keysRedo]
        meanRedoTstat=np.mean([clusterTstats[i] for i in redoClusters])
        corr2,clstrs2,silh2=clusterKMeansTop(corrTmp, maxNumClusters=corrTmp.shape[1]-1,n_init=n_init)
        # Make new outputs, if necessary
        corrNew,clstrsNew,silhNew=makeNewOutputs(corr0, {i:clstrs[i] for i in list(clstrs.keys()) if i not in redoClusters}, clstrs2 )
        newTstatMean=np.mean([np.mean(silhNew[clstrs2[i]])/np.std(silhNew[clstrs2[i]]) for i in list(clstrs2.keys())])
        if newTstatMean<=meanRedoTstat:
            return corr1,clstrs,silh
        else:
            return corrNew,clstrsNew,silhNew


def clusterSpectral(feature, n_clusters = 7):

    feature =((1-feature.fillna(0))/2.)**.5
    mod = SpectralClustering(n_clusters = n_clusters, eigen_solver = 'arpack', random_state = 10, n_init = 100, n_jobs = 10)
    res = mod.fit(feature)
    cluster = {}
    assets = feature.columns.values
    for label in res.labels_:
        cluster[label] = list(assets[res.labels_ == label])

    return cluster


def clusterSimple(corr0, threshold):

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



if  __name__ == '__main__':

    lookback_days = 365
    blacklist = [24, 32, 40]
    factor_ids = ['1200000%02d'%i for i in range(1, 40) if i not in blacklist]
    trade_dates = ATradeDate.month_trade_date(begin_date = '2018-01-01')
    for date in trade_dates:
        # start_date = '%d-%02d-01'%(year, month)
        # end_date = '%d-%02d-01'%(year+1, month)
        start_date = (date - datetime.timedelta(lookback_days)).strftime('%Y-%m-%d')
        end_date = date.strftime('%Y-%m-%d')
        print(start_date, end_date)
        # corr0 = load_fund(start_date, end_date)
        corr0 = load_ind(factor_ids, start_date, end_date)
        # corr1, clstrs, silh = clusterKMeansBase(corr0,maxNumClusters=10,n_init=1)
        factor_name = base_ra_index.load()
        # df_fund = base_ra_fund.load()
        # df_fund.index = df_fund.ra_code.astype('int')
        # df_fund = df_fund.set_index('ra_code')
        # factor_name = df_fund.ra_name
        res = None
        while res is None:
            try:
                # res = clusterKMeansTop(corr0, maxNumClusters=10, n_init=1)
                res = clusterKMeansBase(corr0, maxNumClusters=10, n_init=10)
            except:
                pass

        for k,v in res[1].items():
            v = np.array(v).astype('int')
            print(factor_name.loc[v])
        #     factor_name.loc[v].to_csv('data/fund_cluster/cluster_%d.csv'%k, index_label = 'fund_code')
        print()



