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
import pylab
import matplotlib.pyplot as plt



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
