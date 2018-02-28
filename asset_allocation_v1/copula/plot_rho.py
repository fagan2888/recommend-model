#!/usr/bin/python
# coding=utf-8

from pathos import multiprocessing
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from starvine.bvcopula.copula import *
from starvine.bvcopula import bv_plot
from ipdb import set_trace

def load_data(id_, sdate = None, edate = None):
    data = pd.read_csv("data/ra_index_nav.csv", index_col=["ra_date"], parse_dates=True)
    data = data[data.ra_index_id == id_]
    data = data.loc[:, ['ra_nav']]
    data.index = data.index.to_datetime()
    data = data.sort_index()
    data = data.pct_change(20)
    data = data.replace(0.0, np.nan).dropna()*100
    if sdate is not None:
        data = data[data.index >= sdate] 
    if edate is not None:
        data = data[data.index <= edate] 

    return data


def load_factor_data(id_, layer, sdate = None, edate = None):
    data = pd.read_csv("data/barra_stock_factor_layer_nav.csv", index_col=["trade_date"], parse_dates=True)
    data = data[(data.bf_id == id_) & (data.layer == layer)]
    data = data.loc[:, ['nav']]
    data = data.pct_change(20)*100
    if sdate is not None:
        data = data[data.index >= sdate] 
    if edate is not None:
        data = data[data.index <= edate] 

    return data

def generate_data(asset1, asset2, sdate = None, edate = None):
    sh300 = load_data(asset1, sdate, edate)
    zz500 = load_data(asset2, sdate, edate)

    df = pd.merge(sh300, zz500, left_index=True, right_index=True)
    df.columns = [str(asset1), str(asset2)]
    df = df.replace(0, np.nan).dropna()

    # LIMIT = 1000
    # x = df.sh300.values[-LIMIT:]
    # y = df.zz500.values[-LIMIT:]
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    dates = df.index

    '''
    plt.scatter(x, y)
    plt.title('{}-{}'.format(asset1, asset2))
    try:
        plt.savefig('ret/{}{}_ret'.format(str(asset1)[-2:], str(asset2)[-2:]))
    except Exception:
        pass
    '''

    return x, y, dates



def train(x, y):
    '''
    #
    #frank copula(model3) is the only useful model
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
    return 0
    '''
    c = frank_copula.FrankCopula()
    par, state = c.fitMLE(x, y, [2.7])

    # rand_u = np.linspace(1e-9, 1-1e-9, 20)
    # rand_v = np.linspace(1e-9, 1-1e-9, 20)
    # u, v = np.meshgrid(rand_u, rand_v)
    # p = c.pdf(u.flatten(), v.flatten(), *par)

    # print c.pdf(0.1, 0.2, *par)
    # bv_plot.bvContourf(u.flatten(), v.flatten(), p, title=r"Frank, $\theta: %.2f$" % (par[0]), savefig="frank_copula_pdf.png")
    # set_trace()

    return c, par


def plot_rho(asset1, asset2):
    x, y, _ = generate_data(asset1, asset2, edate = '2011-01-01')
    test_x, test_y, dates = generate_data(asset1, asset2, sdate = '2011-01-01')
    # x = np.random.normal(size = len(x))
    # y = x + 0.1
    LENTH = len(x)
    rank_x = rankdata(x)/LENTH
    rank_y = rankdata(y)/LENTH
    # model, par = train(rank_x, rank_y)
    model, par = train(rank_x, rank_y)

    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.ProcessingPool(processes=cpus)
    # rho = [model.kTau(rank_x[i], rank_y[i], *par) for i in range(LENTH)]
    # rho = pool.map(model.kTau(rank_x[i], rank_y[i], *par) for i in range(LENTH))
    paras = np.repeat(par[0], len(test_x))
    # rho = pool.amap(model.pdf, rank_x, rank_y, paras)
    # rho = pool.amap(model.kTau, rank_x, rank_y, paras)
    rho = pool.amap(model.kTau, test_x, test_y, paras)
    rho = rho.get()

    df = pd.DataFrame(data = rho, index = dates, columns=['kTau{}{}'.format(str(asset1)[-2:], str(asset2)[-2:])])
    df = df.replace(np.inf, 1.0)
    df = df.fillna(method = 'pad')
    df = df.abs()
    df = df.rolling(len(df), 60).mean().dropna()
    df = df[df.index >= '2012-07-27']

    ## save DataFrame
    # df.to_csv('kTau/rho{}{}.csv'.format(str(asset1)[-2:], str(asset2)[-2:]), index_label = 'date')

    # df = df[::10]
    print asset1, asset2, par[0]
    df.plot(ylim = [0,1], label = 'kTau')
    mean_rho = df.values.mean()
    plt.hlines(mean_rho, df.index.min(), df.index.max(), 'c', linestyles='--', label = 'mean kTau: {}'.format(mean_rho))
    plt.legend()
    # df.plot()

    ## save plot
    # plt.savefig('kTau/ktau{}{}.pdf'.format(str(asset1)[-2:], str(asset2)[-2:]))

    return df


def plot_sh300_factor_rho(asset, layer):
    factor_name = asset[:2]+asset[-2:]+str(layer)
    sh300 = load_data(120000001)
    factor = load_factor_data(asset, layer)
    df = pd.merge(sh300, factor, left_index=True, right_index=True, how = 'inner')
    df = df.dropna()
    train_df = df[df.index < '2011-01-01']
    test_df = df[df.index >= '2011-01-01']
    x = train_df.iloc[:, 0].values
    y = train_df.iloc[:, 1].values
    test_x = test_df.iloc[:, 0].values
    test_y = test_df.iloc[:, 1].values
    dates = test_df.index

    LENTH = len(x)
    rank_x = rankdata(x)/LENTH
    rank_y = rankdata(y)/LENTH
    model, par = train(rank_x, rank_y)

    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.ProcessingPool(processes=cpus)
    paras = np.repeat(par[0], len(test_x))
    rho = pool.amap(model.kTau, test_x, test_y, paras)
    rho = rho.get()

    f_name = 'kTau_{}{}'.format('01', factor_name)
    df = pd.DataFrame(data = rho, index = dates, columns=[f_name])
    df = df.replace(np.inf, 1.0)
    df = df.fillna(method = 'pad')
    df = df.abs()
    df = df.rolling(len(df), 60).mean().dropna()
    df = df[df.index >= '2012-07-27']
    df.to_csv('kTau/{}.csv'.format(f_name), index_label = 'date')

    return df


def cal_mul_rho():

    asset1 = 120000001
    asset2 = 120000002
    asset3 = 120000013
    asset4 = 120000014
    asset5 = 120000015
    asset6 = 120000009
    asset7 = 120000029
    asset8 = 120000039
    asset9 = 120000025
    asset10= 120000028

    df12 = plot_rho(asset1, asset2)
    df13 = plot_rho(asset1, asset3)
    df14 = plot_rho(asset1, asset4)
    df15 = plot_rho(asset1, asset5)
    df16 = plot_rho(asset1, asset6)
    df17 = plot_rho(asset1, asset7)
    df18 = plot_rho(asset1, asset8)
    df19 = plot_rho(asset1, asset9)
    df110= plot_rho(asset1, asset10)
    df = pd.concat([
        df12, 
        df13, 
        df14, 
        df15, 
        df16, 
        df17, 
        df18, 
        df19, 
        df110
        ], 1)
    df.to_csv('kTau/kTau_sh300_20.csv', index_label = 'date')


def cal_mul_factor_rho():

    assets = [
        "BF.000001",
        "BF.000002",
        "BF.000003",
        "BF.000004",
        "BF.000005",
        "BF.000006",
        "BF.000007",
        "BF.000008",
        "BF.000009",
        "BF.000010",
        "BF.000011",
        "BF.000012",
    ]
    layers = [0, 1, 2, 3, 4]
    df = None
    for asset in assets:
        for layer in layers:
            print asset, layer
            if df is None:
                df = plot_sh300_factor_rho(asset, layer)
            else:
                df = df.join(plot_sh300_factor_rho(asset, layer))
    set_trace()
    df.to_csv("kTau/kTau_sh300_af.csv", index_label = 'date')


def cluster_days():
    df = pd.read_csv('kTau/kTau_sh300_20.csv', index_col = 0, parse_dates = True)
    df = df.fillna(method = 'pad')
    assets = np.array([x[-2:] for x in df.columns])

    asset_cluster_low = []
    asset_cluster_mid = []
    asset_cluster_high = []
    for idx, row in df.iterrows():
        x = row.values.reshape(-1,1)
        # kmeans2 = KMeans(n_clusters=2, random_state=0, n_jobs=cpus).fit(x)
        # res2 = kmeans2.fit(x)
        # score2 = res2.score(x)
        kmeans3 = KMeans(n_clusters=3, random_state=0).fit(x)
        res3 = kmeans3.fit(x)
        score3 = res3.score(x)

        # model = kmeans2 if score2 > score3 else kmeans3
        model = kmeans3
        kTau_cluster = {}
        asset_cluster = {}
        for i in range(model.n_clusters):
            kTau_cluster[rankdata(model.cluster_centers_)[i]] = row.values[model.labels_ == i]
            asset_cluster[rankdata(model.cluster_centers_)[i]] = assets[model.labels_ == i]
        print idx
        for i in sorted(rankdata(model.cluster_centers_)):
            print i, kTau_cluster[i], asset_cluster[i]
        asset_cluster_low.append('_'.join(asset_cluster[1.0]))
        asset_cluster_mid.append('_'.join(asset_cluster[2.0]))
        asset_cluster_high.append('_'.join(asset_cluster[3.0]))
    
    # set_trace()
    df['low_corr'] = asset_cluster_low
    df['mid_corr'] = asset_cluster_mid
    df['high_corr'] = asset_cluster_high
    df.to_csv('cluster/cluster_result_2.csv', index_label = 'date')

def cluster_all():
    df = pd.read_csv('kTau/kTau_sh300_20.csv', index_col = 0, parse_dates = True)
    df = df.fillna(method = 'pad')

    ## add barra factor
    df2 = pd.read_csv('kTau/kTau_sh300_af.csv', index_col=0, parse_dates=True)
    df2 = df2.fillna(method = 'pad')
    df = pd.merge(df, df2, left_index = True, right_index = True)
    df.to_csv('cluster/cluster_result_3.csv')
    assets = np.array([x[-4:] for x in df.columns])

    x = df.values.T
    model = KMeans(n_clusters=3, random_state=0).fit(x)
    res = model.fit(x)
    score = res.score(x)

    asset_cluster = {}
    for i in range(model.n_clusters):
        asset_cluster[rankdata(model.cluster_centers_.mean(1))[i]] = assets[model.labels_ == i]
    for i in sorted(rankdata(model.cluster_centers_.mean(1))):
        print i, asset_cluster[i]

    

if __name__ == '__main__':
    # cal_mul_rho()
    # cluster()
    cluster_all()

    # load_factor_data("BF.000002", 0)
    # plot_sh300_factor_rho("BF.000002", 0)
    # cal_mul_factor_rho()