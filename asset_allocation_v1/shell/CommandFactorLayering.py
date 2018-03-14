#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

from ipdb import set_trace
import datetime
import json
import os
import sys
sys.path.append('shell')
import click
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
from starvine.bvcopula.copula import frank_copula

from db import database, asset_barra_stock_factor_layer_nav, base_ra_index_nav, base_ra_index, asset_fl_info, asset_fl_nav
from sqlalchemy import MetaData, Table, select, func, literal_column
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
@click.pass_context
def fl(ctx):
    '''
    factor layereing
    '''
    if ctx.invoked_subcommand is None:
        ctx.invoke(fl_update)
        ctx.invoke(fl_nav_update)
    else:
        pass

@fl.command()
@click.pass_context
def fl_update(ctx):
    df_old = asset_fl_info.load()

    factor_layer = FactorLayer(8, 2, 1, base_pool_only = False)
    factor_layer.handle()

    fl_id = []
    fl_asset_id = []
    fl_first_loc = []
    fl_second_loc = []

    layer = 1
    for k, v in factor_layer.l_pool_cluster.iteritems():
        for asset_id in v:
            fl_id.append('FL.00{:02d}{:02d}'.format(1, layer))
            fl_asset_id.append(asset_id)
            fl_first_loc.append(1)
            fl_second_loc.append(int(layer))
        layer += 1

    layer = 1
    for k, v in factor_layer.m_pool_cluster.iteritems():
        for asset_id in v:
            fl_id.append('FL.00{:02d}{:02d}'.format(2, layer))
            fl_asset_id.append(asset_id)
            fl_first_loc.append(2)
            fl_second_loc.append(int(layer))
        layer += 1

    layer = 1
    for k, v in factor_layer.h_pool_cluster.iteritems():
        for asset_id in v:
            fl_id.append('FL.00{:02d}{:02d}'.format(3, layer))
            fl_asset_id.append(asset_id)
            fl_first_loc.append(3)
            fl_second_loc.append(int(layer))
        layer += 1

    df_new = pd.DataFrame(\
        data = np.column_stack([fl_id, fl_asset_id, fl_first_loc, fl_second_loc]),\
        columns = ['fl_id', 'fl_asset_id', 'fl_first_loc', 'fl_second_loc']
        )

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('fl_info', metadata, autoload=True)

    df_new = df_new.set_index(['fl_id', 'fl_asset_id'])
    database.batch(db, t, df_new, df_old)


@fl.command()
@click.pass_context
def fl_nav_update(ctx):
    df_layer = asset_fl_info.load()
    with click.progressbar(length=len(df_layer.index.levels[0]), label='update nav'.ljust(30)) as bar:
        for fl_id in df_layer.index.levels[0]:
            assets = np.array(df_layer.loc[fl_id].index)
            sumple_df = sumple_update(assets, fl_id)

            df_new = sumple_df.reset_index()
            df_new.columns = ['fl_date', 'fl_nav']
            df_new['fl_id'] = fl_id
            df_new = df_new.set_index(['fl_id', 'fl_date'])
            df_old = asset_fl_nav.load_nav(fl_id)

            db = database.connection('asset')
            metadata = MetaData(bind=db)
            t = Table('fl_nav', metadata, autoload=True)
            database.batch(db, t, df_new, df_old)
            bar.update(1)


class FactorLayer():

    def __init__(self, h_num = 8, m_num = 2, l_num = 1, window = 60, base_pool_only = True):

        self._pool = FactorLayer.get_pool(base_pool_only)
        self.window = window
        #self.baseline = 'BF.000001.1'
        self.baseline = '120000039'
        self.h_pool = []
        self.m_pool = []
        self.l_pool = []

        self.h_num = h_num
        self.m_num = m_num
        self.l_num = l_num

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


    def hml_layer(self):
        asset_cluster_hml = self.cluster(self.df_risk_return, 3, self._pool)
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


    @staticmethod
    def train(arr1, arr2):
        c = frank_copula.FrankCopula()
        ## start itering from 2.7, any value here is OK
        par, state = c.fitMLE(arr1, arr2, [2.7])

        return c, par


    @staticmethod
    def cal_rho(id1, id2, window):
        asset1 = load_nav(id1)
        asset2 = load_nav(id2)
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
        model, par = FactorLayer.train(rank_x, rank_y)
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
        asset1 = load_nav(id1)
        asset2 = load_nav(id2)
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

        asset1 = asset1.rolling(20).sum()
        asset2 = asset2.rolling(20).sum()

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
        #df.to_csv('copula/kTau/kTau_sh300_60.csv', index_label = 'date')
        df.to_csv('copula/corr/corr_mf_{}.csv'.format(self.window), index_label = 'date')

        return df


    @staticmethod
    def cal_risk_return(id_):
        asset = load_nav(id_).pct_change()

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
        df.to_csv('copula/rr/risk_return_60.csv')
        return df


    @staticmethod
    def cluster(df, best_cluster_num, pool = None, sdate = None, edate = None, method = 'kmeans'):

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


    def cal_mean_corr(self):
        corr = {}
        for k,v in factor_layer.h_pool_cluster.iteritems():
            layer_assets = []
            for asset in v:
                tmp_nav = load_nav(asset)
                tmp_ret = tmp_nav.pct_change()
                tmp_ret = tmp_ret.replace(0.0, np.nan).dropna()
                layer_assets.append(tmp_ret)
            layer_df = pd.concat(layer_assets, 1)
            layer_df = layer_df[layer_df.index >= '2012-07-27']
            layer_df = layer_df.dropna()
            corr[k] = layer_df.corr().mean().mean()

        mean_corr = np.mean(corr.values())

        print corr
        print mean_corr

        return mean_corr


    def handle(self):
        ## 按照风险收益比将原始资产聚成高、中、低风险三类资产,并将中风险再分成两类
        self.df_risk_return = pd.read_csv('copula/rr/risk_return_60.csv', index_col = 0, parse_dates = True)
        # self.df_risk_return = self.cal_mul_risk_return()
        self.hml_layer()
        self.mid_layer()

        ## 按照相关性将高风险资产分成六类
        # self.df_rho= pd.read_csv('copula/kTau/kTau_sh300_60.csv', index_col = 0, parse_dates = True)
        # self.df_rho= pd.read_csv('copula/corr/corr_mf_{}.csv'.format(window), index_col = 0, parse_dates = True)
        self.df_rho = self.cal_mul_rhos(method = 'corr')
        self.high_layer()
        self.cal_mean_corr()

        ## 低风险只有货币和短融，无需分类
        self.low_layer()


def load_nav(id_):
    if id_.startswith('BF'):
        factor = id_[:-2]
        layer = int(id_[-1])
        df = asset_barra_stock_factor_layer_nav.load_series(factor, layer)
    else:
        df = base_ra_index_nav.load_series(id_)

    return df


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
        asset_nav = load_nav(asset_id)
        asset_nav.columns = [asset_id]
        asset_navs.append(asset_nav)
    df = pd.concat(asset_navs, 1).fillna(method = 'pad').dropna()
    df = df/df.iloc[0, :]
    x = df.values
    sumple_x = sumple(x)
    sumple_df = pd.DataFrame(data = sumple_x, index = df.index, columns = [fl_id])

    df.columns = assets
    df['sumple'] = sumple_x
    df.to_csv('copula/sumple/{}.csv'.format(fl_id), index_label = 'date')

    return sumple_df


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    setup_logging()
    window = 30
    print 'window is {}'.format(window)
    factor_layer = FactorLayer(8, 2, 1, base_pool_only = False, window = window)
    # print factor_layer._pool
    factor_layer.handle()
    for k,v in factor_layer.h_pool_cluster.iteritems():
        print k, v

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
