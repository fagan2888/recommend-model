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
from scipy.spatial import distance
#from starvine.bvcopula.copula import frank_copula

from db import database, asset_barra_stock_factor_layer_nav, base_ra_index_nav, base_ra_index, base_trade_dates, asset_factor_cluster_nav, base_ra_fund, asset_factor_cluster
from db.asset_factor_cluster import *
from db import asset_factor_cluster
# from sqlalchemy import MetaData, Table, select, func, literal_column
from sqlalchemy import * 
from sqlalchemy.orm import sessionmaker
import DBData
import warnings
import factor_cluster

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
        # ctx.invoke(fc_update, optid)
        ctx.invoke(fc_update_nav, optid = optid)
    else:
        pass

@fc.command()
@click.pass_context
def factor_cluster_fund_pool(ctx):

    #factor_cluster_ids = ['FC.000001.3.1', 'FC.000001.3.2', 'FC.000001.3.3', 'FC.000001.3.4', 'FC.000001.3.5', 'FC.000001.3.6', 
    #        'FC.000001.3.7', 'FC.000001.3.8']

    #factor_cluster_ids = ['FC.000002.3.1', 'FC.000002.3.2', 'FC.000002.3.3', 'FC.000002.3.4', 'FC.000002.3.5', 'FC.000002.3.6', 'FC.000002.3.7']

    # factor_cluster_ids_corr = {'FC.000002.3.1' : 0.85, 'FC.000002.3.2': 0.75, 'FC.000002.3.3':0.75, 'FC.000002.3.4':0.8, 'FC.000002.3.5':0.85, 'FC.000002.3.7':0.7}
    factor_cluster_ids_corr = {'FC.000004.1' :0.85}
    stock_pool_codes = list(base_ra_fund.find_type_fund(1).ra_code.ravel())
    # other_pool_codes = list(base_ra_fund.find_type_fund(4).ra_code.ravel())

    # pool_codes = stock_pool_codes + other_pool_codes
    pool_codes = stock_pool_codes

    factor_cluster.factor_cluster_fund_pool(factor_cluster_ids_corr, pool_codes, 53, 5)

    return



@fc.command()
@click.pass_context
def barra_stock_factor_fund_pool(ctx):

    #factor_cluster.barra_stock_factor_fund_pool(53, 5)
    factor_cluster.barra_stock_factor_fund_pool_duplicated(53, 5)

    return



@fc.command()
@click.option('--id', 'optid', help=u'specify markowitz id')
@click.pass_context
def fc_update(ctx, optid):
    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()

    sql = session.query(asset_factor_cluster.factor_cluster.globalid).filter(asset_factor_cluster.factor_cluster.globalid == optid)
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
        fc = FactorCluster(pool, **argv)
        fc.handle()
        with open('model2', 'wb') as f:
            pickle.dump(fc, f)
        # with open('model2', 'rb') as f:
        #     fc = pickle.load(f)

        session.commit()
        session.close()

        fc_update_json(optid, fc)
        fc_update_struct(optid, fc)


def fc_update_json(optid, fc):

    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()

    fc_json_struct = cal_json(optid, fc)
    fc_json_struct = json.dumps(fc_json_struct)

    session.query(
        asset_factor_cluster.factor_cluster.fc_json_struct
        ).filter(asset_factor_cluster.factor_cluster.globalid == optid).\
            update({asset_factor_cluster.factor_cluster.fc_json_struct:fc_json_struct})

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
    fc_json_struct[str(optid)] = {}
    for i in range(1, 1+int(fc.hml_num)):
        fc_json_struct[optid]['{}.{}'.format(optid, i)] = {}

    for k,v in fc.l_pool_cluster.iteritems():
        fc_json_struct[optid]['{}.1'.format(optid)]['{}.1.{}'.format(optid, int(k))]= list(v)
    for k,v in fc.m_pool_cluster.iteritems():
        fc_json_struct[optid]['{}.2'.format(optid)]['{}.2.{}'.format(optid, int(k))]= list(v)
    for k,v in fc.h_pool_cluster.iteritems():
        fc_json_struct[optid]['{}.3'.format(optid)]['{}.3.{}'.format(optid, int(k))]= list(v)

    return fc_json_struct


@fc.command()
@click.option('--id', 'optid', help=u'specify cluster id')
@click.pass_context
def fc_update_nav(ctx, optid):
    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()

    sql1 = session.query(
        distinct(asset_factor_cluster.factor_cluster.fc_json_struct),
        ).filter(asset_factor_cluster.factor_cluster.globalid == optid)

    sql2 = session.query(
        distinct(barra_stock_factor_valid_factor.trade_date),
        )

    fc_json = json.loads(sql1.all()[0][0])
    fc_json = fc_json['%s'%optid]['%s.3'%optid]
    trade_dates = sorted(base_trade_dates.load_trade_dates().index)
    trade_dates_month = [date[0] for date in sql2.all() if date[0] > datetime.date(2011, 1, 1)]
    layers = sorted(fc_json)

    with click.progressbar(length=len(trade_dates_month), label='update layer nav'.ljust(30)) as bar:
        for date in trade_dates_month:
            print
            print date
            sql3 = session.query(
                barra_stock_factor_valid_factor.bf_layer_id
                ).filter(barra_stock_factor_valid_factor.trade_date == date)
            valid_factor = [factor[0] for factor in sql3.all()]
            fc_json_tmp = copy.deepcopy(fc_json)

            large = fc_json_tmp['%s.3.5'%optid]
            large = [factor for factor in large if factor in valid_factor]
            fc_json_tmp['%s.3.5'%optid] = large

            # mid = fc_json_tmp['FC.000001.3.6']
            # mid = [factor for factor in mid if factor in valid_factor]
            # fc_json_tmp['FC.000001.3.6'] = mid

            small = fc_json_tmp['%s.3.1'%optid]
            small = [factor for factor in small if factor in valid_factor]
            fc_json_tmp['%s.3.1'%optid] = small

            for layer in layers:
                layer_ret = []
                factors = fc_json_tmp[layer]
                if len(factors) != 0:
                    print layer
                if len(factors) == 0:
                    continue

                for factor in factors:
                    if factor.startswith('BF'):
                        factor_nav = asset_barra_stock_factor_layer_nav.load_series_selected(factor, reindex = trade_dates, selected_date = date)
                    else:
                        factor_nav = load_nav_series(factor, reindex = trade_dates)
                    factor_ret = factor_nav.pct_change().dropna()
                    factor_ret = factor_ret.loc[date - datetime.timedelta(365): date + datetime.timedelta(31)]
                    layer_ret.append(factor_ret)

                df_layer_ret = pd.concat(layer_ret, 1)
                df_layer_ret = df_layer_ret.mean(1)
                df_layer_nav = (1 + df_layer_ret).cumprod()
                df_layer_nav = df_layer_nav.to_frame(name = 'nav')
                df_layer_nav.index.name = 'date'
                df_layer_nav = df_layer_nav.reset_index()
                df_layer_nav['globalid'] = optid
                df_layer_nav['fc_cluster_id'] = layer
                df_layer_nav['factor_selected_date'] = date
                df_layer_nav = df_layer_nav.set_index(['globalid', 'fc_cluster_id', 'date', 'factor_selected_date'])

                db = database.connection('asset')
                metadata = MetaData(bind=db)
                t = Table('factor_cluster_nav', metadata, autoload=True)

                df_old = asset_factor_cluster_nav.load_nav(optid, layer, date)
                database.batch(db, t, df_layer_nav, df_old)
            print
            bar.update(1)


@fc.command()
@click.option('--id', 'optid', help=u'specify cluster id')
@click.pass_context
def fc_rolling(ctx, optid):

    engine = database.connection('asset')
    Session = sessionmaker(bind = engine)
    session = Session()

    sql = session.query(asset_factor_cluster.factor_cluster.globalid).filter(asset_factor_cluster.factor_cluster.globalid == optid)
    session.commit()
    session.close()

    if len(sql.all()) == 0:
        logger.info('id {} doesn\'t exist'.format(optid))
        return

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
    trade_dates = DBData.trade_dates(start_date = '2012-01-01', end_date = '2018-01-01')
    for date in trade_dates[::12]:
        tmp_argv = copy.deepcopy(argv)
        tmp_argv['end_date'] = date
        fc = FactorCluster(pool, **tmp_argv)
        fc.handle()
        iter_num = 0
        print 'past score:', fc.past_inner_corr, fc.past_outter_corr
        while fc.past_inner_corr < 0.85:
            tmp_argv['h_num'] += 1
            fc = FactorCluster(pool, **tmp_argv)
            fc.handle()
            print 'past score:', fc.past_inner_corr, fc.past_outter_corr
            iter_num += 1
            if iter_num >= 5:
                break

        print date
        print 'score:', fc.inner_score, fc.outter_score
        for i in np.arange(1, tmp_argv['h_num']+1):
            print fc.h_pool_cluster[i]
        print
        print

    session.commit()
    session.close()


class FactorCluster():

    def __init__(self, pool = None, h_num = 8, m_num = 2, l_num = 1, window = 60, base_pool_only = True, hml_num = 3, end_date = None):

        if pool is None:
            self._pool = FactorCluster.get_pool(base_pool_only)
        else:
            self._pool = pool

        if end_date is not None:
            self.end_date = end_date
        else:
            self.end_date = datetime.datetime.today()

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
    def cal_corr(id1, id2, window, end_date = None):
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
        if end_date is not None:
            end_date = end_date.strftime('%Y-%m-%d')
            df = df[df.index <= end_date]

        return df


    def cal_mul_rhos(self, method = 'corr'):

        if method == 'corr':
            df = None
            with click.progressbar(length=len(self.h_pool), label='cal corr'.ljust(30)) as bar:
                for asset in self.h_pool:
                    # print asset,
                    if df is None:
                        df = self.cal_corr(self.baseline, asset, self.window, self.end_date)
                    else:
                        tmp_df = self.cal_corr(self.baseline, asset, self.window, self.end_date)
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


    def cal_mean_corr_future(self, pool):
        corr = {}
        all_layers = []
        for k,v in pool.iteritems():
            layer_assets = []
            for asset in v:
                tmp_nav = load_nav_series(asset)
                tmp_ret = tmp_nav.pct_change()
                tmp_ret = tmp_ret.replace(0.0, np.nan).dropna()
                layer_assets.append(tmp_ret)
            layer_df = pd.concat(layer_assets, 1)
            layer_df = layer_df[layer_df.index >= self.end_date.strftime('%Y-%m-%d')]
            layer_df = layer_df[layer_df.index <= (self.end_date + datetime.timedelta(90)).strftime('%Y-%m-%d')]
            layer_df = layer_df.dropna()
            layer_df = layer_df.rolling(5).sum()[::5]
            all_layers.append(layer_df.mean(1))
            # corr[k] = layer_df.corr().mean().mean()
            corr[k] = cal_corr(layer_df)

        inner_corr = np.mean(corr.values())

        df_all_layers = pd.concat(all_layers, 1)
        outter_corr = cal_corr(df_all_layers)

        return inner_corr, outter_corr


    def cal_mean_corr(self, pool):
        corr = {}
        all_layers = []
        for k,v in pool.iteritems():
            layer_assets = []
            for asset in v:
                tmp_nav = load_nav_series(asset)
                tmp_ret = tmp_nav.pct_change()
                tmp_ret = tmp_ret.replace(0.0, np.nan).dropna()
                layer_assets.append(tmp_ret)
            layer_df = pd.concat(layer_assets, 1)
            layer_df = layer_df[layer_df.index <= self.end_date.strftime('%Y-%m-%d')]
            layer_df = layer_df[layer_df.index >= (self.end_date - datetime.timedelta(365)).strftime('%Y-%m-%d')]
            layer_df = layer_df.dropna()
            layer_df = layer_df.rolling(5).sum()[::5]
            all_layers.append(layer_df.mean(1))
            # corr[k] = layer_df.corr().mean().mean()
            corr[k] = cal_corr(layer_df)

        inner_corr = np.mean(corr.values())

        df_all_layers = pd.concat(all_layers, 1)
        outter_corr = cal_corr(df_all_layers)

        return inner_corr, outter_corr


    def handle(self):
        ## 按照风险收益比将原始资产聚成高、中、低风险三类资产,并将中风险再分成两类
        self.df_risk_return = self.cal_mul_risk_return()
        self.hml_layer(self.hml_num)
        self.mid_layer()

        ## 按照相关性将高风险资产分成六类

        self.df_rho = self.cal_mul_rhos(method = 'corr')
        self.high_layer()
        self.past_inner_corr, self.past_outter_corr = self.cal_mean_corr(self.h_pool_cluster)
        self.inner_score, self.outter_score = self.cal_mean_corr_future(self.h_pool_cluster)

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

def cal_corr(df):
    if len(df.columns) == 1:
        return 1
    else:
        df1 = df.corr()
        df1 = df1[df1 != 1]
        corr = np.nanmean(df1.values)

        return corr


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

            sr = asset_ra_pool_nav.load_series(
                asset_id, 0, 9, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif prefix == 'FD':

            sr = base_ra_fund_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif prefix == 'RS':

            sr = asset_rs_reshape_nav.load_series(
                asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
        elif prefix == 'IX':

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
