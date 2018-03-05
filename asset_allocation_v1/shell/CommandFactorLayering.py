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
from sklearn.cluster import KMeans
from scipy.spatial import distance
from starvine.bvcopula.copula import frank_copula

from db import asset_barra_stock_factor_layer_nav, base_ra_index_nav
import warnings
warnings.filterwarnings('ignore')

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
    pass

@fl.command()
@click.pass_context
def rho_update():
    pass


def cal_risk_return(id_):
    asset = load_nav(id_).pct_change()

    if asset.index[0].date() > datetime.date(2010, 1, 1):
        logging.error("historical data of asset {} must start before 2010-01-01".format(id_))
        return 0

    asset = asset.replace(0.0, np.nan).dropna()
    asset_name = get_asset_name(id_) 
    asset_risk_return = asset.rolling(60).mean()/asset.rolling(60).std()
    asset_risk_return = asset_risk_return.to_frame(name = asset_name)
    asset_risk_return = asset_risk_return.dropna()

    return asset_risk_return


def cal_mul_risk_return():
    base_pool = []
    for base_id in range(2, 42):
        base_pool.append('12{:07}'.format(base_id))

    bf_pool = []
    for factor_id in range(1, 18):
        for layer in range(5):
            bf_pool.append('BF.{:06}.{}'.format(factor_id, layer))
    pool = base_pool+bf_pool

    df = None
    for asset in pool:
        if df is None:
            df = cal_risk_return(asset)
        else:
            tmp_df = cal_risk_return(asset)
            if tmp_df is 0:
                pass
            else:
                df = df.join(tmp_df)

            # print asset, df.mean()
        
    df = df.dropna()
    df.to_csv('copula/rr/rr_all_60.csv', index_label = 'date')


def cluster_risk_return(sdate = None, edate = None):
    df = pd.read_csv('copula/rr/rr_all_60.csv', index_col = 0, parse_dates = True)
    df = df.fillna(method = 'pad')
    if sdate is not None:
        df = df[df.index >= sdate]
    if edate is not None:
        df = df[df.index <= edate]
    
    df = df.loc[:, ['RA09', 'RA10', 'RA11', 'RA26', 'RA27', 'RA37', 'RA38']]
    assets = df.columns.values
    x = df.values.T

    ## Use BIC to choose the best cluster number
    # ks = range(2, 11)
    # kmeans = [KMeans(n_clusters = i, init="k-means++").fit(x) for i in ks]
    # BIC = [compute_bic(kmeansi,x) for kmeansi in kmeans]
    # best_cluster_num = ks[np.argmin(BIC)]
    # logger.info("Best cluster number: {}".format(best_cluster_num))
    
    best_cluster_num = 2
    model = KMeans(n_clusters=best_cluster_num, random_state=0).fit(x)

    asset_cluster = {}
    for i in range(model.n_clusters):
        asset_cluster[rankdata(model.cluster_centers_.mean(1))[i]] = assets[model.labels_ == i]
    # for i in sorted(rankdata(model.cluster_centers_.mean(1))):
        # print i, asset_cluster[i]
    
    return asset_cluster


def train(arr1, arr2):
    '''
    #
    #frank copula(model3) is the only useful model, because all the other model fail to converge
    #

    c = t_copula.StudentTCopula()
    par, state1 = c.fitMLE(x, y, [0.7, 10])
    print 1,par,state1

    c = gauss_copula.GaussCopula()
    par, state2 = c.fitMLE(x, y, [0.7])
    print 2,par,state2

    c = frank_copula.FrankCopula()
    par, state3 = c.fitMLE(x, y, [2.7])
    print 3,par,state3

    c = gumbel_copula.GumbelCopula()
    par, state4 = c.fitMLE(x, y, [2.7])
    print 4,par,state4

    # CLAYTON CDFS
    c = clayton_copula.ClaytonCopula()
    par, state5 = c.fitMLE(x, y, [2.7])
    print 5,par,state5

    c_90 = clayton_copula.ClaytonCopula(1)
    par, state6 = c.fitMLE(x, y, [2.7])
    print 6,par,state6

    c_180 = clayton_copula.ClaytonCopula(2)
    par, state7 = c.fitMLE(x, y, [2.7])
    print 7,par,state7

    c_270 = clayton_copula.ClaytonCopula(3)
    par, state8 = c.fitMLE(x, y, [2.7])
    print 8,par,state8
    '''
    c = frank_copula.FrankCopula()
    ## start itering from 2.7, any value here is OK
    par, state = c.fitMLE(arr1, arr2, [2.7])

    return c, par


def cal_rho(id1, id2):
    asset1 = load_nav(id1)
    asset2 = load_nav(id2)
    if asset1.index[0].date() > datetime.date(2010, 1, 1):
        logging.error("historical data of asset {} must start before 2010-01-01!".format(id1))
        return 0

    # if asset1.index[-1].date() < datetime.date(2018, 1, 1):
    #     logging.error("historical data of asset {} isn't up to date!".format(id1))
    #     return 0

    if asset2.index[0].date() > datetime.date(2010, 1, 1):
        logging.error("historical data of asset {} must start before 2010-01-01!".format(id2))
        return 0

    # if asset2.index[-1].date() < datetime.date(2018, 1, 1):
    #     logging.error("historical data of asset {} isn't up to date!".format(id2))
    #     return 0

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
    model, par = train(rank_x, rank_y)
    if model is 0:
        logging.error("Model fail to converge!".format(id1))
        return 0


    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.ProcessingPool(process=cpus)
    paras = np.repeat(par[0], len(test_x))
    LENTH2 = len(test_x)
    rank_test_x = rankdata(test_x)/LENTH2
    rank_test_y = rankdata(test_y)/LENTH2
    rho = pool.amap(model.kTau, rank_test_x, rank_test_y, paras)
    rho = rho.get()

    name1 = get_asset_name(id1)
    name2 = get_asset_name(id2)
    col_name = '_'.join(['rho', name1, name2])
    df = pd.DataFrame(data = rho, index = test_dates, columns=[col_name])
    df = df.replace(np.inf, 1.0)*10
    df = df.fillna(method = 'pad')
    # df = df.abs()
    # df = df.rolling(120, 60).mean().dropna()
    df = df.rolling(60).mean().dropna()
    df = df[df.index >= '2012-07-27']

    return df


def cal_mul_rhos():
    base_pool = []
    for base_id in range(2, 42):
        base_pool.append('12{:07}'.format(base_id))

    bf_pool = []
    for factor_id in range(1, 18):
        for layer in range(5):
            bf_pool.append('BF.{:06}.{}'.format(factor_id, layer))
    pool = base_pool+bf_pool

    df = None
    for asset in pool:
        if df is None:
            df = cal_rho('120000001', asset)
        else:
            tmp_df = cal_rho('120000001', asset)
            if tmp_df is 0:
                pass
            else:
                df = df.join(tmp_df)
        # print asset, df.mean()
    
    ##去除标准指数的非交易日数据
    df = df.dropna()
    df.to_csv('copula/kTau/kTau_sh300_60.csv', index_label = 'date')
    return df


def cluster(sdate = None, edate = None, best_cluster_num = 8):
    df = pd.read_csv('copula/kTau/kTau_sh300_60.csv', index_col = 0, parse_dates = True)
    ## 只用前10个Barra因子
    df = df.loc[:, 'rho_RA01_RA02':'rho_RA01_BF024']
    ## 选出债券 
    # df = df.loc[:, ['rho_RA01_RA09', 'rho_RA01_RA10', 'rho_RA01_RA11', 'rho_RA01_RA26', \
    # 'rho_RA01_RA27', 'rho_RA01_RA37', 'rho_RA01_RA38']]
    df = df.fillna(method = 'pad')
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
    
    model = KMeans(n_clusters=best_cluster_num, random_state=0).fit(x)

    asset_cluster = {}
    for i in range(model.n_clusters):
        asset_cluster[rankdata(model.cluster_centers_.mean(1))[i]] = assets[model.labels_ == i]
    # for i in sorted(rankdata(model.cluster_centers_.mean(1))):
        # print i, asset_cluster[i]
    
    return asset_cluster


def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)


def cal_layer(id1, id2):
    rho = cal_rho(id1, id2)
    rho_pool = pd.read_csv('copula/kTau/kTau_sh300.csv', index_col = 0, parse_dates = True)
    rho_name = rho.columns[0]
    if rho_name in rho_pool.columns:
        cluster_pool = cluster() 
        for k,v in cluster_pool.iteritems():
            if rho_name in v:
                return k
        logging.error("asset not in cluster pool!")
        os._exit(2)
    else:
        df = rho_pool.join(rho)
        x = df.values.T
        model = KMeans(n_clusters=3, random_state=0).fit(x)
        score = model.score(x)
        return rankdata(model.cluster_centers_.mean(1))[model.labels_[-1]]

 
    # df = rho_pool.join(rho)


def load_nav(id_):
    if id_.startswith('BF'):
        factor = id_[:-2]
        layer = int(id_[-1])
        df = asset_barra_stock_factor_layer_nav.load_series(factor, layer)
    else:
        df = base_ra_index_nav.load_series(id_)

    return df


def get_asset_name(id_):
    if id_.startswith('BF'):
        _, factor, layer = id_.split('.')
        name = 'BF'+factor[-2:]+layer
    else:
        name = 'RA'+id_[-2:]

    return name


def get_asset_id(name):
    if name.startswith("BF"):
        id_ = "BF.0000{}.{}".format(name[-3:-1], name[-1])
    elif name.startswith("RA"):
        id_ = "1200000{}".format(name[-2:])

    return id_


def layer_sumple(sdate):
    asset_cluster = cluster()
    layer_num = len(asset_cluster)
    layer_sumple = []
    for layer in np.arange(1.0, layer_num+1.0):
        assets = asset_cluster[layer]
        asset_ids = [get_asset_id(asset_name.split('_')[-1]) for asset_name in assets]
        df = None
        for asset_id in asset_ids:
            # print asset_id, load_nav(asset_id).index[-1]
            if df is None:
                df = load_nav(asset_id)
                if df.index[-1].date() < datetime.date(2018, 1, 1):
                    logger.warning("data of asset {} is not up to date! Skip.".format(asset_id)) 
                    continue
                df = df.to_frame(name = get_asset_name(asset_id))
            else:
                tmp_df = load_nav(asset_id)
                if tmp_df.index[-1].date() < datetime.date(2018, 1, 1):
                    logger.warning("data of asset {} is not up to date! Skip.".format(asset_id)) 
                    continue
                tmp_df = tmp_df.to_frame(name = get_asset_name(asset_id))
                df = pd.merge(df, tmp_df, left_index = True, right_index = True, how = 'inner')

        df = df[df.index >= sdate]
        df = df/df.values[0]
        tmp_sumple = sumple(df.values)
        sumple_df = pd.DataFrame(data = tmp_sumple, index = df.index, columns = ['lay{}'.format(int(layer))])
        layer_sumple.append(sumple_df)
    
    df = pd.concat(layer_sumple, 1)
    df.columns = ["mfund", "gold", "crude", "hsi", "sp", "bond", "a", "barra"]
    ## Barra factor is not up to date, so we temporarily drop it!
    # df = df.iloc[:, :-1].dropna()
    df = df.dropna()
    df.to_csv('copula/sumple/sumple_result_plus_barra.csv', index_label = 'date')


def sumple(X, maxiter = 100):
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


def sumple_cluster(best_cluster_num = 3):
    df = pd.read_csv('copula/sumple/sumple_result_plus_barra.csv', index_col = 0, parse_dates = True)
    df = df.pct_change()
    df = df.replace(0.0, np.nan).dropna()
    df = df.rolling(60).mean()/df.rolling(60).std()
    df = df.dropna()
    # df.to_csv('copula/sumple/sumple_risk_return.csv', index_label = 'date')

    layers = df.columns.values
    x = df.values.T
    model = KMeans(n_clusters=best_cluster_num, random_state=0).fit(x)

    asset_cluster = {}
    for i in range(model.n_clusters):
        asset_cluster[rankdata(model.cluster_centers_.mean(1))[i]] = layers[model.labels_ == i]
    for i in sorted(rankdata(model.cluster_centers_.mean(1))):
        print i, asset_cluster[i]



if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    setup_logging()
    id1 = '120000010'
    '''
    # id2 = 'BF.000001.0'
    id2 = '120000013'
    # df1 = load_nav(id1)
    # df2 = load_nav(id2)
    # cal_rho(id1, id2)
    # cal_mul_rhos()
    layer = cal_layer(id1, id2)
    logger.info("kTau correlation of {} and {} is in layer {}".format(id1, id2, layer))
    '''
    # cal_mul_rhos()
    # cluster('2016-01-01')
    # cluster()

    # cal_risk_return(id1)
    # cal_mul_risk_return()
    # cluster_risk_return()

    layer_sumple('2010-01-01')
    sumple_cluster()