#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

from ipdb import set_trace
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
from scipy.spatial import distance
#from starvine.bvcopula.copula import frank_copula

from db import database, asset_barra_stock_factor_layer_nav, base_ra_index_nav, base_ra_index, base_trade_dates, asset_factor_cluster_nav
from db.asset_factor_cluster import *
# from sqlalchemy import MetaData, Table, select, func, literal_column
from sqlalchemy import * 
from sqlalchemy.orm import sessionmaker
import warnings
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
def fc(ctx, optid):
    '''
    factor layereing
    '''
    if ctx.invoked_subcommand is None:
        ctx.invoke(fc_update, optid)
    else:
        pass

@fc.command()
@click.option('--id', 'optid', help=u'specify markowitz id')
@click.pass_context
def fc_update(ctx, optid):
    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()

    sql = session.query(factor_cluster.globalid).filter(factor_cluster.globalid == optid)
    session.commit()
    session.close()

    if len(sql.all()) == 0:
        logger.info('id {} doesn\'t exist'.format(optid))
    else:
        engine = database.connection('asset')
        Session = sessionmaker(bind = engine)
        session = Session()

        sql1 = session.query(
            factor_cluster_argv.fc_key,
            factor_cluster_argv.fc_value,
            ).filter(factor_cluster_argv.globalid == optid)

        sql2 = session.query(
            factor_cluster_asset.fc_asset_id
            ).filter(factor_cluster_asset.globalid == optid)

        pool = [asset[0] for asset in sql2.all()]
        argv = {k:int(v) for k,v in sql1.all()}
        # fc = FactorCluster(pool, **argv)
        # fc.handle()
        # with open('model', 'wb') as f:
        #     pickle.dump(fc, f)
        with open('model', 'rb') as f:
            fc = pickle.load(f)

        session.commit()
        session.close()

        #fc_update_json(optid, fc)
        #fc_update_struct(optid, fc)
        fc_update_nav(optid, fc)


def fc_update_json(optid, fc):

    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()

    fc_json_struct = cal_json(optid, fc)

    session.query(
        factor_cluster.fc_json_struct
        ).filter(factor_cluster.globalid == optid).\
            update({factor_cluster.fc_json_struct:str(fc_json_struct)})

    session.commit()
    session.close()


def fc_update_struct(optid, fc):
    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()

    fc_json_struct = cal_json(optid, fc)
    k_v = []
    depth = 0
    def json_kv(dic_json, depth):
        if isinstance(dic_json, dict):
            depth += 1
            for key in dic_json:
                if isinstance(dic_json[key], dict):
                    k_v.append((key, dic_json[key].keys(), depth))
                    # print key, dic_json[key].keys()
                    json_kv(dic_json[key], depth)
                else:
                    k_v.append((key, dic_json[key], depth))
                    # print key, dic_json[key]
                    #dic[key] = dic_json[key]

    json_kv(fc_json_struct, depth)

    fcs = factor_cluster_struct()
    fcs.globalid = optid
    set_trace()
    fcs.fc_parent_cluster_id = -1
    fcs.fc_subject_asset_id = optid
    fcs.depth = 0
    session.merge(fcs)

    for k,v,d in k_v:
        print k,v,d
        for vv in v:
            fcs = factor_cluster_struct()
            fcs.globalid = optid
            fcs.fc_parent_cluster_id = k
            fcs.fc_subject_asset_id = vv
            fcs.depth = d
            session.merge(fcs)
    session.commit()
    session.close()


def cal_json(optid, fc):

    fc_json_struct = {}
    fc_json_struct[optid] = {}
    for i in range(1, 1+int(fc.hml_num)):
        fc_json_struct[optid]['{}.{}'.format(optid, i)] = {}

    for k,v in fc.l_pool_cluster.iteritems():
        fc_json_struct[optid]['{}.1'.format(optid)]['{}.1.{}'.format(optid, int(k))]= list(v)
    for k,v in fc.m_pool_cluster.iteritems():
        fc_json_struct[optid]['{}.2'.format(optid)]['{}.2.{}'.format(optid, int(k))]= list(v)
    for k,v in fc.h_pool_cluster.iteritems():
        fc_json_struct[optid]['{}.3'.format(optid)]['{}.3.{}'.format(optid, int(k))]= list(v)

    return fc_json_struct


def fc_update_nav(optid, fc):
    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()

    sql1 = session.query(
        distinct(factor_cluster_struct.depth),
        )
    depths = [depth[0] for depth in sql1.all()]
    depths = sorted(depths, reverse = True)
    trade_dates = base_trade_dates.load_trade_dates().index
    df_result = pd.DataFrame(columns = ['date', 'nav', 'globalid', 'fc_cluster_id'])

    for depth in depths:
        sql2 = session.query(
            factor_cluster_struct.fc_parent_cluster_id,
            factor_cluster_struct.fc_subject_asset_id,
            ).filter(factor_cluster_struct.depth == depth).statement

        df_depth = pd.read_sql(sql2, session.bind, index_col = ['fc_parent_cluster_id'])
        parents = df_depth.index.unique()
        for parent in parents:
            print parent
            children = df_depth[df_depth.index == parent].values.ravel()
            df = []
            for child in children:
                tmp_df = load_nav_series(asset_id = child, reindex = trade_dates, begin_date = '2010-01-01').sort_index()
                df.append(tmp_df)
            df = pd.concat(df, 1)
            df = df.dropna()
            df = df.pct_change()
            df = df.fillna(0.0)
            df = df.mean(1)
            df = df + 1
            df = df.cumprod()

            df = df.to_frame(name = 'nav')
            df.index.name = 'date'
            df = df.reset_index()
            df['globalid'] = optid
            df['fc_cluster_id'] = parent
            df_result = pd.concat([df_result, df])

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('factor_cluster_nav', metadata, autoload=True)

    df_old = asset_factor_cluster_nav.load_all(optid)
    df_old = df_old.set_index(['globalid', 'fc_cluster_id', 'date'])
    df_result = df_result.set_index(['globalid', 'fc_cluster_id', 'date'])
    database.batch(db, t, df_result, df_old)
    set_trace()


class FactorCluster():

    def __init__(self, pool = None, h_num = 8, m_num = 2, l_num = 1, window = 60, base_pool_only = True, hml_num = 3):

        if pool is None:
            self._pool = FactorCluster.get_pool(base_pool_only)
        else:
            self._pool = pool

        self.window = window
        #self.baseline = 'BF.000001.1'
        self.baseline = '120000039'
        self.hml_num = hml_num

        self.score = 0
        self.h_pool = []
        self.m_pool = []
        self.l_pool = []

        self.h_num = h_num
        self.m_num = m_num
        self.l_num = l_num

        self.asset_cluster_hml = {}
        self.h_pool_cluster = {}
        self.m_pool_cluster = {}
        self.l_pool_cluster = {}


    @staticmethod
    def get_pool(base_pool_only):

        '''
        base_pool = []
        for base_id in range(2, 42):
            base_pool.append('12{:07}'.format(base_id))

        bf_pool = []
        for factor_id in range(1, 3):
            for layer in range(5):
                bf_pool.append('BF.{:06}.{}'.format(factor_id, layer))
        pool = base_pool+bf_pool
        '''

        base_pool = base_ra_index.load_globalid()
        base_pool = base_pool.values
        base_pool = [str(x) for x in base_pool]
        base_pool = np.sort(base_pool).tolist()
        if base_pool_only:
            return base_pool

        bf_pool = []
        bf_pool_df = asset_barra_stock_factor_layer_nav.load_layer_id()
        for idx, row in bf_pool_df.iterrows():
            bf_pool.append('{}.{}'.format(row['bf_id'], row['layer']))

        ## drop '12000001' from base_pool because it it the baseline
        pool = base_pool + bf_pool

        return pool


    def hml_layer(self, hml_num):
        asset_cluster_hml = self.cluster(self.df_risk_return, hml_num, self._pool, method = 'kmeans')
        self.asset_cluster_hml = asset_cluster_hml
        self.h_pool = asset_cluster_hml[1.0]
        self.m_pool = asset_cluster_hml[2.0]
        self.l_pool = asset_cluster_hml[3.0]


    def mid_layer(self):
        self.m_pool_cluster = self.cluster(self.df_risk_return, self.m_num, self.m_pool)


    def high_layer(self):
        self.h_pool_cluster = self.cluster(self.df_rho, self.h_num, self.h_pool, method = 'agg')
        '''
        for k, v in self.h_pool_cluster.iteritems():
            if 'BF.00000' in v:
                v = np.insert(v, 0, self.baseline)
                self.h_pool_cluster[k] = v
                '''


    def low_layer(self):
        self.l_pool_cluster[1.0] = self.l_pool


    #@staticmethod
    #def train(arr1, arr2):
    #    c = frank_copula.FrankCopula()
    #    ## start itering from 2.7, any value here is OK
    #    par, state = c.fitMLE(arr1, arr2, [2.7])

    #    return c, par


    @staticmethod
    def cal_rho(id1, id2, window):
        asset1 = load_nav_series(id1)
        asset2 = load_nav_series(id2)
        if asset1.index[0].date() > datetime.date(2010, 1, 5):
            logger.warning("historical data of asset {} must start before 2010-01-05!".format(id1))
            return 0

        if asset1.index[-1].date() < datetime.date(2018, 1, 1):
            logger.warning("historical data of asset {} isn't up to date!".format(id1))
            return 0

        if asset2.index[0].date() > datetime.date(2010, 1, 5):
            logger.warning("historical data of asset {} must start before 2010-01-05!".format(id2))
            return 0

        if asset2.index[-1].date() < datetime.date(2018, 1, 1):
            logger.warning("historical data of asset {} isn't up to date!".format(id2))
            return 0

        dates1 = asset1.index
        dates2 = asset2.index
        joint_dates = dates1.intersection(dates2)
        asset1 = asset1.reindex(joint_dates)
        asset2 = asset2.reindex(joint_dates)
        asset1 = asset1.pct_change(20).dropna()*100
        asset2 = asset2.pct_change(20).dropna()*100

        x = asset1[asset1.index <= '2011-01-01']
        y = asset2[asset1.index <= '2011-01-01']
        test_x = asset1[asset1.index > '2011-01-01']
        test_y = asset2[asset1.index > '2011-01-01']
        test_dates = test_x.index

        LENTH1 = len(x)
        rank_x = rankdata(x)/LENTH1
        rank_y = rankdata(y)/LENTH1
        model, par = FactorCluster.train(rank_x, rank_y)
        if model is 0:
            logger.warning("Model fail to converge!".format(id1))
            return 0


        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.ProcessingPool(process=cpus)
        paras = np.repeat(par[0], len(test_x))
        LENTH2 = len(test_x)
        rank_test_x = rankdata(test_x)/LENTH2
        rank_test_y = rankdata(test_y)/LENTH2
        rho = pool.amap(model.kTau, rank_test_x, rank_test_y, paras)
        rho = rho.get()

        df = pd.DataFrame(data = rho, index = test_dates, columns=[id2])
        df = df.replace(np.inf, 1.0)*10
        df = df.fillna(method = 'pad')
        # df = df.abs()
        # df = df.rolling(120, 60).mean().dropna()
        df = df.rolling(window).mean().dropna()
        df = df[df.index >= '2012-07-27']

        return df


    @staticmethod
    def cal_corr(id1, id2, window):
        asset1 = load_nav_series(id1)
        asset2 = load_nav_series(id2)
        if asset1.index[0].date() > datetime.date(2010, 1, 5):
            logger.warning("historical data of asset {} must start before 2010-01-05!".format(id1))
            return 0

        if asset1.index[-1].date() < datetime.date(2018, 1, 1):
            logger.warning("historical data of asset {} isn't up to date!".format(id1))
            return 0

        if asset2.index[0].date() > datetime.date(2010, 1, 5):
            logger.warning("historical data of asset {} must start before 2010-01-05!".format(id2))
            return 0

        if asset2.index[-1].date() < datetime.date(2018, 1, 1):
            logger.warning("historical data of asset {} isn't up to date!".format(id2))
            return 0

        asset1 = asset1.pct_change().dropna()
        asset2 = asset2.pct_change().dropna()
        asset1 = asset1.replace(0.0, np.nan).dropna()
        asset2 = asset2.replace(0.0, np.nan).dropna()

        asset1 = asset1.rolling(5).sum()
        asset2 = asset2.rolling(5).sum()

        dates1 = asset1.index
        dates2 = asset2.index
        joint_dates = dates1.intersection(dates2)
        asset1 = asset1.reindex(joint_dates)
        asset2 = asset2.reindex(joint_dates)

        df = pd.rolling_corr(asset1, asset2, window).dropna()
        df = df.to_frame(name = id2)
        df = df[df.index >= '2010-01-04']

        return df


    def cal_mul_rhos(self, method = 'corr'):

        if method == 'corr':
            df = None
            with click.progressbar(length=len(self.h_pool), label='cal corr'.ljust(30)) as bar:
                for asset in self.h_pool:
                    # print asset,
                    if df is None:
                        df = self.cal_corr(self.baseline, asset, self.window)
                    else:
                        tmp_df = self.cal_corr(self.baseline, asset, self.window)
                        if tmp_df is 0:
                            pass
                        else:
                            df = df.join(tmp_df)
                    bar.update(1)
            # print asset, df.mean()

        elif method == 'rho':
            df = None
            with click.progressbar(length=len(self.h_pool), label='cal corr'.ljust(30)) as bar:
                for asset in self.h_pool:
                    if df is None:
                        df = self.cal_rho(self.baseline, asset, self.window)
                    else:
                        tmp_df = self.cal_rho(self.baseline, asset, self.window)
                        if tmp_df is 0:
                            pass
                        else:
                            df = df.join(tmp_df)
                    bar.update(1)

        ##去除标准指数的非交易日数据
        df = df.dropna()

        return df


    @staticmethod
    def cal_risk_return(id_):
        asset = load_nav_series(id_).pct_change()

        if asset.index[0].date() > datetime.date(2010, 1, 5):
            logger.warning("historical data of asset {} must start before 2010-01-05".format(id_))
            return 0

        if asset.index[-1].date() < datetime.date(2018, 1, 1):
            logger.warning("data of asset {} isn't up to date".format(id_))
            return 0

        asset = asset.replace(0.0, np.nan).dropna()
        asset_risk_return = asset.rolling(60).mean()/asset.rolling(60).std()
        asset_risk_return = asset_risk_return.to_frame(name = id_)
        asset_risk_return = asset_risk_return.dropna()

        return asset_risk_return


    def cal_mul_risk_return(self):
        df = None
        for asset in self._pool:
            if df is None:
                df = self.cal_risk_return(asset)
            else:
                tmp_df = self.cal_risk_return(asset)
                if tmp_df is 0:
                    pass
                else:
                    df = df.join(tmp_df)
        return df


    @staticmethod
    def cluster(df, best_cluster_num, pool = None, sdate = None, edate = None, method = 'agg'):

        if pool is not None:
            try:
                pool = df.columns.intersection(pool)
            except:
                set_trace()
            df = df.loc[:, pool]
        df = df.fillna(method = 'pad')
        df = df.dropna()
        if sdate is not None:
            df = df[df.index >= sdate]
        if edate is not None:
            df = df[df.index <= edate]

        assets = df.columns.values
        x = df.values.T

        ## Use BIC to choose the best cluster number
        # ks = range(2, 11)
        # kmeans = [KMeans(n_clusters = i, init="k-means++").fit(x) for i in ks]
        # BIC = [compute_bic(kmeansi,x) for kmeansi in kmeans]
        # best_cluster_num = ks[np.argmin(BIC)]
        # logger.info("Best cluster number: {}".format(best_cluster_num))

        if method == 'kmeans':
            model = KMeans(n_clusters=best_cluster_num, random_state=0).fit(x)
            asset_cluster = {}
            for i in range(model.n_clusters):
                asset_cluster[rankdata(model.cluster_centers_.mean(1))[i]] = assets[model.labels_ == i]
            # for i in sorted(rankdata(model.cluster_centers_.mean(1))):
                # print i, asset_cluster[i]

        elif method == 'agg':
            model = AgglomerativeClustering(n_clusters=best_cluster_num, linkage='ward').fit(x)
            asset_cluster = {}
            for i in np.arange(model.n_clusters):
                asset_cluster[i+1.0] = assets[model.labels_ == i]

        return asset_cluster


    def cal_mean_corr(self, pool):
        corr = {}
        for k,v in pool.iteritems():
            layer_assets = []
            for asset in v:
                tmp_nav = load_nav_series(asset)
                tmp_ret = tmp_nav.pct_change()
                tmp_ret = tmp_ret.replace(0.0, np.nan).dropna()
                layer_assets.append(tmp_ret)
            layer_df = pd.concat(layer_assets, 1)
            layer_df = layer_df[layer_df.index >= '2012-07-27']
            layer_df = layer_df.dropna()
            layer_df = layer_df.rolling(5).sum()[::5]
            corr[k] = layer_df.corr().mean().mean()

        mean_corr = np.mean(corr.values())

        # print corr
        # print mean_corr

        return mean_corr


    def handle(self):
        ## 按照风险收益比将原始资产聚成高、中、低风险三类资产,并将中风险再分成两类
        self.df_risk_return = self.cal_mul_risk_return()
        self.hml_layer(self.hml_num)
        self.mid_layer()

        ## 按照相关性将高风险资产分成六类

        self.df_rho = self.cal_mul_rhos(method = 'corr')
        self.high_layer()
        self.score = self.cal_mean_corr(self.h_pool_cluster)

        ## 低风险只有货币和短融，无需分类
        self.low_layer()


# def load_nav(id_):
#     if id_.startswith('BF'):
#         factor = id_[:-2]
#         layer = int(id_[-1])
#         df = asset_barra_stock_factor_layer_nav.load_series(factor, layer)
#     else:
#         df = base_ra_index_nav.load_series(id_)

#     return df


def load_nav_series(asset_id, reindex=None, begin_date=None, end_date=None):

    prefix = asset_id[0:2]
    if prefix.isdigit():
        xtype = int(asset_id) / 10000000
        if xtype == 1:
            #
            # 基金池资产
            #
            asset_id = int(asset_id) % 10000000
            (pool_id, category) = (asset_id / 100, asset_id % 100)
            ttype = pool_id / 10000
            sr = asset_ra_pool_nav.load_series(
                pool_id, category, ttype, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif xtype == 3:
            #
            # 基金池资产
            #
            sr = base_ra_fund_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif xtype == 4:
            #
            # 修型资产
            #
            sr = asset_rs_reshape_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif xtype == 12:
            #
            # 指数资产
            #
            sr = base_ra_index_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif xtype == 'ERI':

            sr = base_exchange_rate_index_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        else:
            sr = pd.Series()
    else:
        if prefix == 'AP':
            #
            # 基金池资产
            #
            sr = asset_ra_pool_nav.load_series(
                asset_id, 0, 9, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif prefix == 'FD':
            #
            # 基金资产
            #
            sr = base_ra_fund_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif prefix == 'RS':
            #
            # 修型资产
            #
            sr = asset_rs_reshape_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif prefix == 'IX':
            #
            # 指数资产
            #
            sr = base_ra_index_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif prefix == 'ER':

            sr = base_exchange_rate_index_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)

        elif prefix == 'FC':

            sr = asset_factor_cluster_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)

        elif prefix == 'BF':

            sr = asset_barra_stock_factor_layer_nav.load_series_2(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        else:
            sr = pd.Series()

    return sr


def sumple(X, maxiter = 100):
    if X.shape[1] == 1:
        return X.ravel()
    ncor, n = X.shape
    X = hilbert(X ,axis = 0)
    X = mat(X)
    x = X[:ncor, :]
    i = 1
    weight = mat(np.ones((1,n)))
    while i <= maxiter:
        i += 1
        #weight = np.mean(np.conj(x.dot(weight.conj().T).dot(np.ones((1,n))) - \
        #        x.dot(np.diag(weight[0])))*x,0).reshape(1,-1)
        #weight = (np.sqrt(n/(weight.dot(weight.conj().T))))*weight
        weight = np.mean(np.multiply((x*weight.conj().T*np.ones((1,n)) - \
                x*np.diag(np.array(weight.conj())[0])).conj(), x),0)
        w = weight*weight.conj().T
        w_real = np.real(w[0,0])
        weight = np.sqrt(n/w_real)*weight
    s = (X.dot(weight.conj().T))/np.sum(np.abs(weight))
    r = np.real(s)
    r = np.array(r).flat[:]
    return r


def sumple_update(assets, fl_id):
    asset_navs = []
    for asset_id in assets:
        asset_nav = load_nav_series(asset_id)
        asset_nav.columns = [asset_id]
        asset_navs.append(asset_nav)
    df = pd.concat(asset_navs, 1).fillna(method = 'pad').dropna()
    df = df/df.iloc[0, :]
    x = df.values
    sumple_x = sumple(x)
    sumple_df = pd.DataFrame(data = sumple_x, index = df.index, columns = [fl_id])

    df.columns = assets
    df['sumple'] = sumple_x

    return sumple_df


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    setup_logging()
    window = 240
    print 'window is {}'.format(window)
    with open('copula/score/score_window_{}.csv'.format(window), 'wb') as f:
        f.write('layer_num, score\n')
        for i in range(2, 15):
            factor_cluster = FactorCluster(i, 2, 1, base_pool_only = False, window = window)
            factor_cluster.handle()
            score = factor_cluster.score
            f.write('{}, {}\n'.format(i, score))
    # print factor_layer._pool
    # factor_layer.handle()
    # for k,v in factor_layer.h_pool_cluster.iteritems():
    #     print k, v

    # print equity_pool
    '''
    equity_nav = []
    for asset in equity_pool:
        asset_nav = load_nav(asset).loc['2011-01-01':'2018-03-01']
        equity_nav.append(asset_nav)
        asset_nav.name = asset
        print asset, len(asset_nav)
    equity_nav_df = pd.concat(equity_nav, 1)
    '''
