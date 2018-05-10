#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

from ipdb import set_trace
import copy
import datetime
import json
import os
import sys
sys.path.append('shell')
import click
import pickle
import pandas as pd
import numpy as np
import logging
import logging.config
from numba import jit, double
from numpy import mat
from pathos import multiprocessing
from scipy.stats import rankdata
from scipy.signal import hilbert
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial import distance, distance_matrix
#from starvine.bvcopula.copula import frank_copula

from db import database, asset_barra_stock_factor_layer_nav, base_ra_index_nav, base_ra_index, base_trade_dates, asset_factor_cluster_nav, base_ra_fund, asset_factor_cluster, base_trade_dates
from db.asset_factor_cluster import *
from db import asset_factor_cluster
# from sqlalchemy import MetaData, Table, select, func, literal_column
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
import DBData
import warnings
import factor_cluster
from CommandFactorCluster import load_nav_series
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def setup_logging(
    default_path = './shell/logging.json',
    default_level = logging.INFO,
    env_key = 'LOG_CFG'):

    """Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)



@click.group(invoke_without_command=True)
@click.option('--id', 'optid', help=u'specify markowitz id')
@click.pass_context
def fcn(ctx, optid):
    '''
    factor layereing
    '''
    if ctx.invoked_subcommand is None:
        # ctx.invoke(fc_update, optid)
        ctx.invoke(fc_update_nav, optid = optid)
    else:
        pass


@fcn.command()
@click.option('--id', 'optid', help=u'specify cluster id')
@click.pass_context
def fc_rolling(ctx, optid):

    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()


    sql1 = session.query(
        factor_cluster_asset.fc_asset_id
        ).filter(factor_cluster_asset.globalid == optid)
    asset_ids = [asset[0] for asset in sql1.all()]
    assets = {}
    for asset_id in asset_ids:
        assets[asset_id] = load_nav_series(asset_id)

    # for asset_id in ['120000001', '120000003', '120000013']:
    #     assets[asset_id] = load_nav_series(asset_id)


    sql2 = session.query(
        distinct(barra_stock_factor_valid_factor.trade_date)
        )
    select_dates = [select_date[0] for select_date in sql2.all()]
    select_dates = select_dates[20:]

    sql3 = session.query(
        barra_stock_factor_valid_factor.trade_date,
        barra_stock_factor_valid_factor.bf_layer_id,
        ).statement
    valid_factors = pd.read_sql(sql3, session.bind, index_col = ['trade_date'], parse_dates = True)


    trade_dates = DBData.trade_dates(start_date = '2012-01-01', end_date = '2018-01-01')
    # bar = click.progressbar(length = len(trade_dates), label = 'Factor Cluster'.ljust(30))
    # for date in trade_dates[::4]:
    layer_result = {}
    layer_result['date'] = []
    layer_result['layer'] = []
    layer_result['factor'] = []

    lookback_days = 365*5
    forecast_days = 90

    # f1 = open('inner_corr.csv', 'wb')
    # f2 = open('outter_corr.csv', 'wb')
    # f1.write('date, inner score, outter score\n')
    # f2.write('date, finner score, foutter score\n')
    f1 = open('inner_beta.csv', 'wb')
    f2 = open('outter_beta.csv', 'wb')
    f1.write('date, inner beta, outter beta\n')
    f2.write('date, finner beta, foutter beta\n')

    df_result = pd.DataFrame(columns = ['date', 'factor_id', 'layer'])
    non_bf_assets = [asset for asset in asset_ids if not asset.startswith('BF')]
    for date in select_dates:
        print date
        sdate = (date - datetime.timedelta(lookback_days)).strftime('%Y-%m-%d')
        # sdate = (select_dates[-1] - datetime.timedelta(lookback_days)).strftime('%Y-%m-%d')
        edate = date.strftime('%Y-%m-%d')
        # edate = select_dates[-1].strftime('%Y-%m-%d')
        fdate = (date + datetime.timedelta(forecast_days)).strftime('%Y-%m-%d')
        bf_ids = valid_factors.loc[date].values.ravel()
        used_ids = np.union1d(bf_ids, non_bf_assets)

        '''
        init_num = 5
        fc = FactorCluster(assets, init_num, sdate, edate, fdate)
        fc.handle()
        while fc.inner_score < 0.88:
            init_num += 1
            fc = FactorCluster(assets, init_num, sdate, edate, fdate)
            fc.handle()
        '''

        method = 'corr'
        scores = {}
        models = {}
        # bf_ids = None
        for i in range(2, 10):
        # for i in [4]:
            # fc = FactorCluster(assets, i, sdate, edate, fdate, method = method, bf_ids = used_ids)
            fc = FactorCluster(assets, i, sdate, edate, fdate, method = method, bf_ids = None)
            fc.handle()
            print i, 'silhouette_samples_value:', fc.silhouette_samples_value
            score = fc.silhouette_samples_value
            # score = fc.inner_score - fc.outter_score
            # score = fc.outter_score + i/20.0
            # if fc.inner_score < 0.85:
                # score = score + 10
            # if i == 1 and fc.inner_score > 0.85:
                # score = -1
            # score = fc.outter_ret - fc.inner_ret
            # score = fc.inner_score
            # print i, fc.inner_score, fc.outter_score, score
            # print i, fc.inner_ret, fc.outter_ret, fc.inner_ret - fc.outter_ret
            scores[score] = i
            # if score > 0.85:
                # break
            models[score] = fc

        best_score = np.max(scores.keys())
        # best_cluster_num = scores[best_score]
        best_model = models[best_score]
        fc = best_model
        # best_cluster_num = 12

        # if best_model.silhouette_samples_value < 0.75:

        # if best_model.outter_score > 0.20:
        #     fc = FactorCluster(assets, 1, sdate, edate, fdate, method = method, bf_ids = used_ids)
        #     fc.handle()
        # else:
        #     fc = best_model

        # fc = FactorCluster(assets, best_cluster_num, sdate, edate, fdate, method = method, bf_ids = bf_ids)

        print date, fc.layer_ret.values
        # bar.update(1)

        print 'best cluster num:', fc.n_clusters
        print 'train:', fc.inner_score, fc.outter_score
        print 'test:', fc.finner_score, fc.foutter_score

        # print 'train:', fc.inner_ret, fc.outter_ret
        # print 'test:', fc.finner_ret, fc.foutter_ret

        f1.write('%s, %s, %s\n'%(date, fc.inner_score, fc.outter_score))
        f2.write('%s, %s, %s\n'%(date, fc.finner_score, fc.foutter_score))
        # f1.write('%s, %s, %s\n'%(date, fc.inner_ret, fc.outter_ret))
        # f2.write('%s, %s, %s\n'%(date, fc.finner_ret, fc.foutter_ret))

        # print date,',',fc.finner_score,',',fc.foutter_score

        # count = 0
        for k, v in fc.asset_cluster.iteritems():

        #     if v[0].startswith('BF'):
        #         count +=1
        #         for factor in v:
        #             layer_result['date'].append(date)
        #             layer_result['layer'].append(count)
        #             layer_result['factor'].append(factor)

            print k, v
            for vv in v:
                df_result.loc[len(df_result)] = [date, vv, k]

        # print 'layer of barra:', count
        print

    f1.close()
    f2.close()
    df_result = df_result.set_index('date')
    df_result.to_csv('data/df_result.csv', index_label = 'date')

    # df_layer_result = pd.DataFrame(layer_result)
    # df_layer_result = df_layer_result.set_index('date')
    # df_layer_result.to_csv('df_layer_result.csv', index_label = 'date')
    # set_trace()

    session.commit()
    session.close()


@fcn.command()
@click.option('--id', 'optid', help=u'specify cluster id')
@click.pass_context
def fc_update_nav(ctx, optid):

    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()

    sql1 = session.query(
        factor_cluster_asset.fc_asset_id
        ).filter(factor_cluster_asset.globalid == optid)
    asset_ids = [asset[0] for asset in sql1.all()]
    assets = {}
    for asset_id in asset_ids:
        assets[asset_id] = load_nav_series(asset_id)

    start_date = '2010-01-01'
    end_date = '2018-12-01'
    # trade_dates = DBData.trade_dates(start_date = '2010-01-01', end_date = '2018-12-01')
    trade_dates = base_trade_dates.load_trade_dates(begin_date = '2010-01-01', end_date = '2018-12-01').index
    df_assets = pd.DataFrame(assets)
    df_assets = df_assets[df_assets.index >= start_date]
    df_assets = df_assets[df_assets.index <= end_date]
    df_assets = df_assets.reindex(trade_dates).dropna()
    df_ret = df_assets.pct_change().dropna()

    df_layer_result = pd.read_csv('df_layer_result.csv', index_col = 0, parse_dates = True)
    dates = df_layer_result.index.drop_duplicates()
    df_new = pd.DataFrame(columns = ['date', 'nav', 'fc_cluster_id', 'factor_selected_date'])
    for date in dates:
        print date
        df = df_layer_result.loc[date]

        layers = np.unique(df.layer)
        df_layer = pd.DataFrame(columns = ['date', 'nav', 'fc_cluster_id'])
        for layer in layers:
            factors = df[df.layer == layer]
            factors = factors.factor.values
            factors_ret = df_ret.loc[:, factors]
            layer_ret = factors_ret.mean(1)
            layer_nav = (1 + layer_ret).cumprod()
            layer_nav = layer_nav.to_frame(name = 'nav')
            layer_nav.index.name = 'date'
            layer_nav['fc_cluster_id'] = optid + '.%d'%layer
            layer_nav = layer_nav.reset_index()
            df_layer = pd.concat([df_layer, layer_nav])

        df_layer['factor_selected_date'] = date
        df_new = pd.concat([df_new, df_layer])

    df_new['globalid'] = optid
    df_new = df_new.set_index(['globalid', 'fc_cluster_id', 'date', 'factor_selected_date'])

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('factor_cluster_nav', metadata, autoload=True)

    df_old = asset_factor_cluster_nav.load_nav(optid)
    database.batch(db, t, df_new, df_old)


@fcn.command()
@click.option('--id', 'optid', help=u'specify cluster id')
@click.pass_context
def fc_update_bflayer_nav(ctx, optid):

    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()

    sql1 = session.query(
        factor_cluster_asset.fc_asset_id
        ).filter(factor_cluster_asset.globalid == optid)
    asset_ids = [asset[0] for asset in sql1.all()]
    assets = {}
    for asset_id in asset_ids:
        print asset_id
        assets[asset_id] = load_nav_series(asset_id)

    start_date = '2010-01-01'
    # end_date = '2018-12-01'
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    # trade_dates = DBData.trade_dates(start_date = '2010-01-01', end_date = '2018-12-01')
    # trade_dates = base_trade_dates.load_trade_dates(begin_date = '2010-01-01', end_date = '2018-12-01').index
    trade_dates = base_trade_dates.load_trade_dates(begin_date = '2010-01-01', end_date = end_date).index
    df_assets = pd.DataFrame(assets)
    df_assets = df_assets[df_assets.index >= start_date]
    df_assets = df_assets[df_assets.index <= end_date]
    df_assets = df_assets.reindex(trade_dates).dropna()
    df_ret = df_assets.pct_change().dropna()

    sql2 = session.query(
        distinct(barra_stock_factor_valid_factor.trade_date)
        )
    select_dates = [select_date[0] for select_date in sql2.all()]
    select_dates = select_dates[20:]

    sql3 = session.query(
        barra_stock_factor_valid_factor.trade_date,
        barra_stock_factor_valid_factor.bf_layer_id,
        ).statement
    valid_factors = pd.read_sql(sql3, session.bind, index_col = ['trade_date'], parse_dates = True)

    df_new = pd.DataFrame(columns = ['date', 'nav', 'fc_cluster_id', 'factor_selected_date'])
    for date in select_dates:
        print date
        bf_ids = valid_factors.loc[date].values.ravel()
        factors_ret = df_ret.loc[:, bf_ids]
        layer_ret = factors_ret.mean(1)
        layer_nav = (1 + layer_ret).cumprod()
        layer_nav = layer_nav.to_frame(name = 'nav')
        layer_nav.index.name = 'date'
        layer_nav['fc_cluster_id'] = '%s.%d'%(optid, 2)
        layer_nav = layer_nav.reset_index()

        layer_nav['factor_selected_date'] = date
        df_new = pd.concat([df_new, layer_nav])

    df_new['globalid'] = optid
    df_new = df_new.set_index(['globalid', 'fc_cluster_id', 'date', 'factor_selected_date'])

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('factor_cluster_nav', metadata, autoload=True)

    df_old = asset_factor_cluster_nav.load_nav(optid)
    database.batch(db, t, df_new, df_old)


class FactorCluster(object):

    def __init__(self, assets, n_clusters, start_date, end_date, fdate, method = 'corr', bf_ids = None):
        self.n_clusters = n_clusters
        self.method = method
        self.bf_ids = bf_ids
        self.ret = self.cal_asset_ret(assets, start_date, end_date)
        self.fret = self.cal_asset_ret(assets, end_date, fdate)


    def cal_asset_ret(self, assets, start_date, end_date):
        trade_dates = DBData.trade_dates(start_date = '2010-01-01', end_date = '2018-12-01')
        df_assets = pd.DataFrame(assets)
        df_assets = df_assets[df_assets.index >= start_date]
        df_assets = df_assets[df_assets.index <= end_date]
        df_assets = df_assets.reindex(trade_dates).dropna()
        if self.bf_ids is not None:
            df_assets = df_assets.loc[:, self.bf_ids]
        # df_ret = np.log(df_assets).diff().dropna()
        # df_ret = np.exp(df_ret*52) - 1
        # df_ret = df_assets/df_assets.iloc[0]
        df_ret = df_assets.pct_change().dropna()
        #df_ret = np.exp(df_ret * 52)

        # df_ret.to_csv('df_ret.csv', index_label = 'date')
        # sys.exit(0)

        # return df_ret
        return df_ret


    def train(self, ret, n_clusters, method = 'corr'):

        asset_names = ret.columns
        if method == 'corr':
            distance_matrix = pearson_affinity
        if method == 'adj_corr':
            distance_matrix = adjusted_pearson_affinity_ori
        if method == 'beta':
            distance_matrix = adjusted_pearson_affinity_std
        elif method == 'lp':
            distance_matrix = lp_distance

        cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity=distance_matrix)
        # print ret.T
        cluster.fit(ret.T)
        asset_cluster = {}
        for i in np.arange(n_clusters):
            asset_cluster[i+1] = asset_names[cluster.labels_ == i]

        if n_clusters >= 2:
            silhouette_samples_value = silhouette_score(distance_matrix(ret.T.values), cluster.labels_, 'precomputed')
            self.silhouette_samples_value = silhouette_samples_value
        else:
            self.silhouette_samples_value = 1

        return asset_cluster


    def score(self):
        if self.method == 'corr':
            cal_score = cal_corr
        if self.method == 'adj_corr':
            cal_score = cal_adj_corr
        if self.method == 'beta':
            cal_score = cal_std_corr
        elif self.method == 'lp':
            cal_score = cal_lp

        inner_score = []
        all_nav_ret = []
        for layer in self.asset_cluster.keys():
            layer_assets = self.asset_cluster[layer]
            layer_asset_ret = self.ret.loc[:, layer_assets]
            # try:
            #     print layer_asset_ret.corr(method = 'spearman').loc['BF.000001.0', 'BF.000001.1']
            # except:
            #     pass
            # inner_score.append(cal_score(layer_asset_ret, method = 'outter'))
            inner_score.append(cal_score(layer_asset_ret))
            layer_nav_ret = layer_asset_ret.mean(1)
            all_nav_ret.append(layer_nav_ret)

        df_all_nav = pd.concat(all_nav_ret, 1)
        # outter_score = cal_score(df_all_nav, method = 'outter')
        outter_score = cal_score(df_all_nav)
        # print np.mean(inner_score), outter_score

        return np.mean(inner_score), outter_score


    def ret_diff(self):

        inner_score = []
        all_nav_ret = []
        for layer in self.asset_cluster.keys():
            layer_assets = self.asset_cluster[layer]
            layer_asset_ret = self.ret.loc[:, layer_assets]
            # try:
            #     print layer_asset_ret.corr(method = 'spearman').loc['BF.000001.0', 'BF.000001.1']
            # except:
            #     pass
            inner_score.append(cal_ret(layer_asset_ret))
            layer_nav_ret = layer_asset_ret.mean(1)
            all_nav_ret.append(layer_nav_ret)

        df_all_nav = pd.concat(all_nav_ret, 1)
        outter_score = cal_ret(df_all_nav)
        # print np.mean(inner_score), outter_score

        return np.mean(inner_score), outter_score


    def fscore(self):
        if self.method == 'corr':
            cal_score = cal_corr
        if self.method == 'adj_corr':
            cal_score = cal_adj_corr
        if self.method == 'beta':
            cal_score = cal_std_corr
        elif self.method == 'lp':
            cal_score = cal_lp

        inner_score = []
        all_nav_ret = []
        for layer in self.asset_cluster.keys():
            layer_assets = self.asset_cluster[layer]
            layer_asset_ret = self.fret.loc[:, layer_assets]
            # try:
            #     print layer_asset_ret.corr(method = 'spearman').loc['BF.000001.0', 'BF.000001.1']
            # except:
            #     pass
            # inner_score.append(cal_score(layer_asset_ret, method = 'outter'))
            inner_score.append(cal_score(layer_asset_ret))
            layer_nav_ret = layer_asset_ret.mean(1)
            all_nav_ret.append(layer_nav_ret)

        df_all_nav = pd.concat(all_nav_ret, 1)
        # outter_score = cal_score(df_all_nav, method = 'outter')
        outter_score = cal_score(df_all_nav)
        # print np.mean(inner_score), outter_score

        return np.mean(inner_score), outter_score


    def fret_diff(self):

        inner_score = []
        all_nav_ret = []
        for layer in self.asset_cluster.keys():
            layer_assets = self.asset_cluster[layer]
            layer_asset_ret = self.fret.loc[:, layer_assets]
            # try:
            #     print layer_asset_ret.corr(method = 'spearman').loc['BF.000001.0', 'BF.000001.1']
            # except:
            #     pass
            inner_score.append(cal_ret(layer_asset_ret))
            layer_nav_ret = layer_asset_ret.mean(1)
            all_nav_ret.append(layer_nav_ret)

        df_all_nav = pd.concat(all_nav_ret, 1)
        outter_score = cal_ret(df_all_nav)
        self.layer_ret = df_all_nav.mean()
        # print np.mean(inner_score), outter_score

        return np.mean(inner_score), outter_score


    def handle(self):
        self.asset_cluster = self.train(self.ret, self.n_clusters, method = self.method)
        self.inner_score, self.outter_score = self.score()
        self.finner_score, self.foutter_score = self.fscore()
        self.inner_ret, self.outter_ret = self.ret_diff()
        self.finner_ret, self.foutter_ret = self.fret_diff()


def adjusted_pearson_affinity_ori(M):
    # print np.array([[pearsonr(a,b)[0] for a in M] for b in M])
    result = np.zeros((len(M), len(M)))
    for i in range(len(M)):
        for j in range(len(M)):
            a = M[i]
            b = M[j]
            # length_a = np.linalg.norm(a)
            # length_b = np.linalg.norm(b)
            length_a = np.prod(1+a)
            length_b = np.prod(1+b)
            # print length_a, length_b
            if length_a > length_b:
                scale = length_a/length_b
            else:
                scale = length_b/length_a

            result[i][j] = pearsonr(a,b)[0]/scale

    return 1 - np.array(result)


def adjusted_pearson_affinity_std(M):
    # print np.array([[pearsonr(a,b)[0] for a in M] for b in M])
    result = np.zeros((len(M), len(M)))
    for i in range(len(M)):
        for j in range(len(M)):
            a = M[i]
            b = M[j]
            # length_a = np.linalg.norm(a)
            # length_b = np.linalg.norm(b)
            length_a = np.std(a)
            length_b = np.std(b)
            # print length_a, length_b
            if length_a > length_b:
                scale = length_a/length_b
            else:
                scale = length_b/length_a

            ret_a = (1+a).cumprod()
            ret_b = (1+b).cumprod()
            result[i][j] = pearsonr(ret_a,ret_b)[0]/scale

    return 1 - np.array(result)


@jit
def adjusted_pearson_affinity(M):
    length = len(M)
    if length == 1:
        return np.array([1.0])

    result = []
    for i in range(length):
        result.append([])
        for j in range(length):
            result[i].append(0)

    for i in range(length):
        for j in range(length):
            a = M[i]
            b = M[j]
            # length_a = np.linalg.norm(a)
            # length_b = np.linalg.norm(b)
            nav_a = 1
            nav_b = 1
            for k in a:
                nav_a *= k+1
            for k in b:
                nav_b *= k+1
            # print length_a, length_b
            if nav_a > nav_b:
                scale = nav_a/nav_b
            else:
                scale = nav_b/nav_a

            # result[i][j] = pearsonr(a,b)[0]/scale
            mean_a = 0
            mean_b = 0
            corr_a = 0
            corr_b = 0
            cov_ab = 0
            pearson_ab = 0
            for k in a:
                mean_a += k
            for k in b:
                mean_b += k
            mean_a = mean_a/length
            mean_b = mean_b/length

            length_a = len(a)
            for k in a:
                corr_a += (k - mean_a)**2
            for k in b:
                corr_b += (k - mean_b)**2
            corr_a = corr_a/length_a
            corr_b = corr_b/length_a
            std_a = corr_a**0.5
            std_b = corr_b**0.5

            for k in range(length_a):
                cov_ab += (a[k] - mean_a)*(b[k] - mean_b)
            cov_ab = cov_ab/length_a
            pearson_ab = cov_ab/(std_a*std_b)
            result[i][j] = 1 - pearson_ab/scale

    return result


def pearson_affinity(M):

    return 1 - np.array([[pearsonr(a,b)[0] for a in M] for b in M])


def lp_distance(x):

    return distance_matrix(x, x)


def cal_corr(df):
    if len(df.columns) == 1:
        return 1

    df1 = df.corr()
    df1 = df1[df1 != 1]
    corr = np.nanmean(df1)

    return corr


def cal_adj_corr(df):
    df = df.T
    if len(df.index) == 1:
        return 1

    df1 = df.values
    # df1 = 1 - np.array(adjusted_pearson_affinity(df1))
    df1 = 1 - np.array(adjusted_pearson_affinity_ori(df1))
    df1 = df1[df1 != 1]
    corr = np.nanmean(df1)

    return corr


def cal_std_corr(df, method = 'outter'):
    df = df.T
    if len(df.index) == 1:
        return 1

    df1 = df.values
    # df1 = 1 - np.array(adjusted_pearson_affinity(df1))
    df1 = 1 - np.array(adjusted_pearson_affinity_std(df1))
    df1 = df1[df1 != 1]

    if method == 'inner':
        corr = np.nanmin(df1)
    elif method == 'outter':
        corr = np.nanmean(df1)

    return corr


def cal_lp(df):
    if len(df.columns) == 1:
        return 0

    df1 = distance_matrix(df.T, df.T)
    df1 = df1[df1 != 0]
    lp = np.nanmean(df1)

    return lp


def cal_ret(df):
    length = len(df.columns)
    if length == 1:
        return 0

    ret = (df + 1).prod().values
    ret = ret**(52.0/len(df))

    result = 0
    for i in ret:
        result += np.abs(ret - i).sum()

    result = result/(length*length - length)

    return result


def cal_layer_type():
    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()

    sql1 = session.query(
        factor_cluster_asset.fc_asset_id
        ).filter(factor_cluster_asset.globalid == 'FC.000004')
    asset_ids = [asset[0] for asset in sql1.all()]
    assets = {}
    for asset_id in asset_ids:
        assets[asset_id] = load_nav_series(asset_id)

    trade_dates = DBData.trade_dates(start_date = '2010-06-01', end_date = '2018-12-01')
    df_assets = pd.DataFrame(assets)
    # df_assets = df_assets[df_assets.index >= start_date]
    # df_assets = df_assets[df_assets.index <= end_date]
    df_assets = df_assets.reindex(trade_dates).dropna()
    df_ret = df_assets.pct_change().dropna()

    layer_info = pd.read_csv('data/df_result.csv', index_col = 0, parse_dates = True)
    dates = layer_info.index.unique()

    sh300 = base_ra_index_nav.load_series('120000001')
    sh300 = sh300.reindex(trade_dates).pct_change().dropna()
    zz500 = base_ra_index_nav.load_series('120000002')
    zz500 = zz500.reindex(trade_dates).pct_change().dropna()
    cyb = base_ra_index_nav.load_series('120000018')
    cyb = cyb.reindex(trade_dates).pct_change().dropna()

    for i in range(12, len(dates)):
        start_date = dates[i-12]
        end_date = dates[i]
        layers = layer_info.loc[end_date].layer.unique()
        tmp_sh300 = sh300.loc[start_date.date():end_date.date()]
        tmp_zz500 = zz500.loc[start_date.date():end_date.date()]
        tmp_cyb = cyb.loc[start_date.date():end_date.date()]
        for layer in layers:
            layer_factor_id = layer_info[layer_info.layer == layer].loc[end_date].factor_id
            if type(layer_factor_id) == str:
                layer_factor_id = np.array([layer_factor_id])
            else:
                layer_factor_id = layer_factor_id.values

            layer_ret = df_ret.loc[start_date.date():end_date.date(), layer_factor_id]
            layer_ret = layer_ret.mean(1)
            # layer_nav = (1+layer_ret.mean(1)).cumprod()
            sh300_corr = np.corrcoef(layer_ret, tmp_sh300)[0][1]
            zz500_corr = np.corrcoef(layer_ret, tmp_zz500)[0][1]
            cyb_corr = np.corrcoef(layer_ret, tmp_cyb)[0][1]
            corr_list = [sh300_corr, zz500_corr, cyb_corr]
            max_corr = max(corr_list)
            if max_corr < 0.85:
                sign = 'unknown'
            elif sh300_corr == max_corr:
                sign = 'sh300'
            elif zz500_corr == max_corr:
                sign = 'zz500'
            elif cyb_corr == max_corr:
                sign = 'cybzs'

            print end_date, layer, sign, sh300_corr, zz500_corr, cyb_corr


if __name__ == '__main__':

    cal_layer_type()
