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
from numpy import mat
from pathos import multiprocessing
from scipy.stats import rankdata
from scipy.signal import hilbert
from sklearn.cluster import KMeans, AgglomerativeClustering
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


    sql2 = session.query(
        distinct(barra_stock_factor_valid_factor.trade_date)
        )
    select_dates = [select_date[0] for select_date in sql2.all()]
    select_dates = select_dates[20:]


    sql1 = session.query(
        factor_cluster_asset.fc_asset_id
        ).filter(factor_cluster_asset.globalid == optid)
    asset_ids = [asset[0] for asset in sql1.all()]
    assets = {}
    for asset_id in asset_ids:
        assets[asset_id] = load_nav_series(asset_id)


    trade_dates = DBData.trade_dates(start_date = '2012-01-01', end_date = '2018-01-01')
    # bar = click.progressbar(length = len(trade_dates), label = 'Factor Cluster'.ljust(30))
    # for date in trade_dates[::4]:
    layer_result = {}
    layer_result['date'] = []
    layer_result['layer'] = []
    layer_result['factor'] = []

    for date in select_dates:
        print date
        sdate = (date - datetime.timedelta(365)).strftime('%Y-%m-%d')
        edate = date.strftime('%Y-%m-%d')
        fdate = (date + datetime.timedelta(90)).strftime('%Y-%m-%d')

        '''
        init_num = 5
        fc = FactorCluster(assets, init_num, sdate, edate, fdate)
        fc.handle()
        while fc.inner_score < 0.88:
            init_num += 1
            fc = FactorCluster(assets, init_num, sdate, edate, fdate)
            fc.handle()
        '''

        scores = {}
        for i in range(5, 10):
            fc = FactorCluster(assets, i, sdate, edate, fdate)
            fc.handle()
            score = fc.inner_score - fc.outter_score
            # score = -fc.outter_score
            scores[score] = i

        best_score = np.max(scores.keys())
        best_cluster_num = scores[best_score]
        # best_cluster_num = 7
        fc = FactorCluster(assets, best_cluster_num, sdate, edate, fdate)
        fc.handle()
        # bar.update(1)

        print best_cluster_num
        print fc.inner_score, fc.outter_score
        # print date,',',fc.finner_score,',',fc.foutter_score

        count = 0
        for k, v in fc.asset_cluster.iteritems():
            if v[0].startswith('BF'):
                count +=1
                for factor in v:
                    layer_result['date'].append(date)
                    layer_result['layer'].append(count)
                    layer_result['factor'].append(factor)
            print k, v
        print 'layer of barra:', count
        print

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


class FactorCluster(object):

    def __init__(self, assets, n_clusters, start_date, end_date, fdate, method = 'exp'):
        self.ret = self.cal_asset_ret(assets, start_date, end_date)
        self.fret = self.cal_asset_ret(assets, end_date, fdate)
        self.n_clusters = n_clusters
        self.method = method


    def cal_asset_ret(self, assets, start_date, end_date):
        trade_dates = DBData.trade_dates(start_date = '2010-01-01', end_date = '2018-12-01')
        df_assets = pd.DataFrame(assets)
        df_assets = df_assets[df_assets.index >= start_date]
        df_assets = df_assets[df_assets.index <= end_date]
        df_assets = df_assets.reindex(trade_dates).dropna()
        # df_ret = np.log(df_assets).diff().dropna()
        # df_ret = np.exp(df_ret*52) - 1
        # df_assets = df_assets/df_assets.iloc[0]
        df_ret = df_assets.pct_change().dropna()
        #df_ret = np.exp(df_ret * 52)

        # return df_ret
        return df_ret


    @staticmethod
    def train(ret, n_clusters, method = 'exp'):

        def pearson_affinity(M):
            # print np.array([[pearsonr(a,b)[0] for a in M] for b in M])
            return 1 - np.array([[pearsonr(a,b)[0] for a in M] for b in M])

        def lp_distance(x):

            return distance_matrix(x, x)

        asset_names = ret.columns
        if method == 'corr':
            cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity=pearson_affinity)
        elif method == 'lp':
            cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity=pearson_affinity)
        elif method == 'exp':
            ret = np.exp(ret * 13)
            cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity=pearson_affinity)

        cluster.fit(ret.T)
        asset_cluster = {}
        for i in np.arange(n_clusters):
            asset_cluster[i+1] = asset_names[cluster.labels_ == i]

        return asset_cluster


    def score(self):
        inner_score = []
        all_nav_ret = []
        for layer in self.asset_cluster.keys():
            layer_assets = self.asset_cluster[layer]
            layer_asset_ret = self.ret.loc[:, layer_assets]
            # try:
            #     print layer_asset_ret.corr(method = 'spearman').loc['BF.000001.0', 'BF.000001.1']
            # except:
            #     pass
            inner_score.append(cal_corr(layer_asset_ret))
            layer_nav_ret = layer_asset_ret.mean(1)
            all_nav_ret.append(layer_nav_ret)

        df_all_nav = pd.concat(all_nav_ret, 1)
        outter_score = cal_corr(df_all_nav)

        return np.mean(inner_score), outter_score


    def fscore(self):
        inner_score = []
        all_nav_ret = []
        for layer in self.asset_cluster.keys():
            layer_assets = self.asset_cluster[layer]
            layer_asset_ret = self.fret.loc[:, layer_assets]
            # try:
            #     print layer_asset_ret.corr(method = 'spearman').loc['BF.000001.0', 'BF.000001.1']
            # except:
            #     pass
            inner_score.append(cal_corr(layer_asset_ret))
            layer_nav_ret = layer_asset_ret.mean(1)
            all_nav_ret.append(layer_nav_ret)

        df_all_nav = pd.concat(all_nav_ret, 1)
        outter_score = cal_corr(df_all_nav)

        return np.mean(inner_score), outter_score


    def handle(self):
        self.asset_cluster = FactorCluster.train(self.ret, self.n_clusters, method = self.method)
        self.inner_score, self.outter_score = self.score()
        self.finner_score, self.foutter_score = self.fscore()



def cal_corr(df):
    if len(df.columns) == 1:
        return 1

    df1 = df.corr()
    df1 = df1[df1 != 1]
    corr = np.nanmean(df1)

    return corr


if __name__ == '__main__':

    pass
