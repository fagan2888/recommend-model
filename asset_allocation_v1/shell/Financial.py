#coding=utf8


import string
import math
from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import *
from sklearn import datasets, linear_model
import numpy as np
from scipy.stats import norm
import cvxopt
from cvxopt import matrix, solvers
from numpy import isnan
from scipy import linalg
#import pylab
#import matplotlib.pyplot as plt


def efficient_frontier_spe(return_rate, bound, sum1 = 0.65, sum2 = 0.45):

    solvers.options['show_progress'] = False

    n_asset    =     len(return_rate)

    asset_mean = np.mean(return_rate, axis = 1)
    #print asset_mean
    l = len(asset_mean)

    cov        =     np.cov(return_rate)

    S           =     matrix(cov)
    l2 = matrix(S * np.eye(l))
    l2 = l2 * 2
    S = S + l2
    #S = l2

    pbar       =     matrix(asset_mean)

    #
    # 设置限制条件, 闲置条件的意思
    #    GW <= H
    # 其中:
    #    G 是掩码矩阵, 其元素只有0,1,
    #    W 是每个资产的权重,
    #    H 是一个值
    # 对于i行的意思是 sum(Gij * Wi | j=0..n_asset-1) <= Hi
    #
    # 具体地, 本函数有4类闲置条件:
    #    1: Wi >= 0, 由G的0..(n_asset-1) 行控制
    #    2: Wi下限, 由G的n_asset..(2*n_asset-1)行控制
    #    3: Wi的上限,由G的2*n_asset..(3*n_asset-1)行控制
    #    4: 某类资产之和的上限, 由G的3*n_asset 行控制
    #

    G = matrix(0.0, (3 * n_asset + 2,  n_asset))
    h = matrix(0.0, (3 * n_asset + 2, 1) )

    h[3* n_asset, 0] = sum1
    h[3* n_asset + 1, 0] = sum2
    for i in range(0, n_asset):
        #
        # Wi >= 0
        #
        # h[i, 0] = 0
        G[i, i] = -1
        #
        # Wi的下限
        #
        G[n_asset + i, i ]     = -1
        h[n_asset + i, 0] = -1.0 * bound[i]['lower']
        #
        # Wi的上限
        #
        G[2 * n_asset + i, i ] = 1
        h[2 * n_asset + i, 0] = bound[i]['upper']
        #
        # 某类资产之和的上限
        #
        if bound[i]['sum1'] == True or bound[i]['sum1'] == 1:
            G[3 * n_asset, i] = 1

        if bound[i]['sum2'] == True or bound[i]['sum2'] == 1:
            G[3 * n_asset + 1, i] = 1

    A          =  matrix(1.0, (1, n_asset))
    b          =  matrix(1.0)

    N          = 200
    mus        = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
    returns    = [ dot(pbar,x) for x in portfolios ]
    risks      = [ sqrt(dot(x, S*x)) for x in portfolios ]


    return risks, returns, portfolios



def efficient_frontier_spe_bl(return_rate, P, eta, alpha, bound, sum1 = 0.65, sum2 = 0.45):

    solvers.options['show_progress'] = False

    n_asset    =     len(return_rate)

    cov = np.cov(return_rate)

    initialvalue = np.mean(return_rate, axis=1)

    expected_return = black_litterman(initialvalue, cov, P, eta, alpha)

    S = matrix(cov + cov * np.eye(len(expected_return)) * 2)         #Double the diagonal of cov matrix

    pbar       =     matrix(expected_return)

    #
    # 设置限制条件, 闲置条件的意思
    #    GW <= H
    # 其中:
    #    G 是掩码矩阵, 其元素只有0,1,
    #    W 是每个资产的权重,
    #    H 是一个值
    # 对于i行的意思是 sum(Gij * Wi | j=0..n_asset-1) <= Hi
    #
    # 具体地, 本函数有4类闲置条件:
    #    1: Wi >= 0, 由G的0..(n_asset-1) 行控制
    #    2: Wi下限, 由G的n_asset..(2*n_asset-1)行控制
    #    3: Wi的上限,由G的2*n_asset..(3*n_asset-1)行控制
    #    4: 某类资产之和的上限, 由G的3*n_asset 行控制
    #
    G = matrix(0.0, (3 * n_asset + 2,  n_asset))
    h = matrix(0.0, (3 * n_asset + 2, 1) )

    h[3* n_asset, 0] = sum1
    h[3* n_asset + 1, 0] = sum2
    for i in range(0, n_asset):
        #
        # Wi >= 0
        #
        # h[i, 0] = 0
        G[i, i] = -1
        #
        # Wi的下限
        #
        G[n_asset + i, i ]     = -1
        h[n_asset + i, 0] = -1.0 * bound[i]['lower']
        #
        # Wi的上限
        #
        G[2 * n_asset + i, i ] = 1
        h[2 * n_asset + i, 0] = bound[i]['upper']
        #
        # 某类资产之和的上限
        #
        if bound[i]['sum1'] == True or bound[i]['sum1'] == 1:
            G[3 * n_asset, i] = 1

        if bound[i]['sum2'] == True or bound[i]['sum2'] == 1:
            G[3 * n_asset + 1, i] = 1

    A          =  matrix(1.0, (1, n_asset))
    b          =  matrix(1.0)

    N          = 200
    mus        = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
    returns    = [ dot(pbar,x) for x in portfolios ]
    risks      = [ sqrt(dot(x, S*x)) for x in portfolios ]


    return risks, returns, portfolios



#计算下方差
def semivariance(portfolio):
    mean       =        np.mean(portfolio)
    var        =        0.0
    n          =        0
    for d in portfolio:
        if d <= mean:
            n   = n + 1
            var = var + (d - mean) ** 2
    var        =    var / n

    var        =    math.sqrt(var)
    if np.isnan(var) or np.isinf(var):
        var = 0.0
    return     var



#jensen测度
'''
def jensen(portfolio, market, rf):

    pr         =    np.mean(portfolio)
    mr         =    np.mean(market)
    beta       =    np.cov(portfolio, market)[0][1] / np.cov(market)
    return pr - (rf + beta * ( mr - rf) )
'''


def jensen(portfolio, market, rf):

    p = []
    m = []
    for i in range(0, len(portfolio)):
        #print portfolio[i], market[i]
        p.append(portfolio[i] - rf)
        m.append(market[i] - rf)

    p = np.array(p)
    m = np.array(m)

    clf       = linear_model.LinearRegression()
    clf.fit(m.reshape(len(m),1), p.reshape(len(p), 1))
    #clf.fit(m, p)
    alpha = clf.intercept_[0]
    beta  = clf.coef_[0]

    #if np.isnan(alpha) or np.isinf(alpha):
    #    alpha = 0.0

    return alpha



#sharp
def sharp(portfolio, rf):
    r         =    np.mean(portfolio)
    sigma     =    np.std(portfolio)
    sharpe    = (r - rf) /sigma
    if np.isnan(sharpe) or np.isinf(sharpe):
        sharpe = 0.0
    return    sharpe


#sharp
def sharp_annual(portfolio, rf):

    r         =    np.mean(portfolio) * 52
    sigma     =    np.std(portfolio) * (52 ** 0.5)
    rf        =    0.03

    sharpe    = (r - rf) /sigma
    if np.isnan(sharpe) or np.isinf(sharpe):
        sharpe = 0.0
    return    sharpe


#索提诺比率
def sortino(portfolio, rf):
    pr        =    np.mean(portfolio)
    sortino_v = (pr - rf) / semivariance(portfolio)
    if np.isnan(sortino_v) or np.isinf(sortino_v):
        sortino_v = 0.0
    return    sortino_v


#treynor-mazuy模型
def tm(portfolio, market, rf):
    xparams   = []
    yparams   = []
    for i in range(0, len(portfolio)):
        yparams.append(portfolio[i] - rf)
        xparams.append([market[i] - rf, (market[i] - rf) ** 2])

    clf       = linear_model.LinearRegression()
    clf.fit(xparams, yparams)
    return clf.intercept_,  clf.coef_[0], clf.coef_[1]


#henrikson-merton
def hm(portfolio, market, rf):
    xparams   = []
    yparams   = []
    for i in range(0, len(portfolio)):
        yparams.append(portfolio[i] - rf)
        if rf - market[i] > 0:
            xparams.append([market[i] - rf, market[i] - rf])
        else:
            xparams.append([market[i] - rf, 0])

    clf       = linear_model.LinearRegression()
    clf.fit(xparams, yparams)
    return clf.intercept_,  clf.coef_[0], clf.coef_[1]



#value at risk
def var(portfolio):
    parray = np.array(portfolio)
    valueAtRisk = norm.ppf(0.05, parray.mean(), parray.std())
    return valueAtRisk



#positive peroid weight
def ppw(portfolio, benchmark):


    #print 'p', portfolio
    #print 'm', benchmark

    solvers.options['show_progress'] = False

    length = len(benchmark)
    A = []
    b = []

    for i in range(0, length):
        item = []
        for j in range(0, length + 3):
            item.append(0)
        A.append(item)

    for i in range(0,  length + 3):
        b.append(0)

    for i in range(0, length):
        A[i][i] = -1
        b[i]    = -1e-13

    i = length
    for j in range(0, length):
        A[j][i]     = 1
    b[i] = 1


    i = length + 1
    for j in range(0, length):
        A[j][i] = -1
    b[i] = -1


    i = length + 2
    for j in range(0, length):
        A[j][i] = -benchmark[j]

    b[i] = 0

    c = []
    for j in range(0, length):
        c.append(benchmark[j])


    A = matrix(A)
    b = matrix(b)
    c = matrix(c)

    #print A
    #print b
    #print c

    sol = solvers.lp(c, A, b)
    ppw = 0
    for i in range(0, len(sol['x'])):
        ppw = ppw + portfolio[i] * sol['x'][i]

    return ppw



def grs(portfolio):
    return 0



def black_litterman(initialvalue, Sigma, P, eta, alpha):

    #Use the weight coming from risk parity to do reverse optimization
    pi = initialvalue

    var_view = np.dot(np.dot(P, Sigma), P.T)

    Q = [P[k].dot(pi) + eta[k] * np.sqrt(var_view[k,k]) for k in range(P.shape[0])]
    Q = np.array(np.matrix(Q).T)

    Omega = var_view / alpha * np.eye(P.shape[0])

    # Compute posterior estimate of the mean
    # This is a simplified version of formula (8) on page 4.
    middle = linalg.inv(var_view + Omega)
    # Use this line if the argument P and Q are just matrix
    # er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(ts,P.T),middle),(Q - np.dot(P,pi.T)).T)
    # Use this line if the argument P and Q are wrapped matrices (for calculating the Omega by the formula given in the paper)
    er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(Sigma,P.T),middle),(Q - np.expand_dims(np.dot(P,pi.T),axis=1)))
    # The following formula is directly coming from the paper
    # er = np.dot(linalg.inv(linalg.inv(ts)+np.dot(P.T, np.dot(linalg.inv(Omega), P))), np.dot(linalg.inv(ts), pi)+np.dot(P.T, np.dot(linalg.inv(Omega), Q))
    # posteriorcov = Sigma - Sigma.dot(P.T).dot(middle).dot(P).dot(Sigma)
    return er



if __name__ == '__main__':


    rs = [[0.25,0.3,0.4, 0.3, 0.2, -0.1, -0.2], [0.1, 0.2, 0.3, -0.01, -0.2, 0.01, 0.02], [0.001, 0.02, -0.03, 0.05, -0.06, -0.07, 0.1]]
    test_efficient_frontier(rs)

    #print portfolio[0]
    #print np.cov(rs)
           #print semivariance(rs[0])
    #print jensen(rs[0], rs[1], 0.02)
    #print sharp(rs[0], 0.02)
    #print sortino(rs[0], 0.02)
    #print tm(rs[0], rs[1], 0.02)
    #print ppw(rs[0], rs[1])

    '''
    weq = np.array([0.016,0.022,0.052,0.055,0.116,0.124,0.615])
    C = np.array([[ 1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
          [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
          [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
          [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
          [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
          [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
          [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]])
    Sigma = np.array([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187])
    refPi = np.array([0.039, 0.069, 0.084, 0.090, 0.043, 0.068, 0.076])
    assets= {'Australia','Canada   ','France   ','Germany  ','Japan    ','UK       ','USA      '}


    V = np.multiply(np.outer(Sigma, Sigma), C)

    delta = 2.5
    tau = 0.05


    tauV = tau * V
    P1 = np.array([0, 0, -.295, 1.00, 0, -.705, 0 ])
    P2 = np.array([1, 0, 0, 0, 0, -0.5, -0.5])
    Q1 = np.array([0.05])
    Q2 = np.array([0.06])
    P=np.array([P1, P2])
    Q=np.array([Q1, Q2]);

    #print P
    #print Q
    #print np.dot(np.dot(P,tauV),P.T)
    Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])
    #print np.eye(Q.shape[0])
    #print Omega

    res = black_litterman(delta, weq, V, tau, P, Q, Omega)

    #print res

    '''
