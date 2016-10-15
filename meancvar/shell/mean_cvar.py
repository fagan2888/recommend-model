#coding=utf8


import pandas as pd
import numpy  as np
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
import sys
sys.path.append('shell')
import FundValue as fv


def markowitz(dfr):

    solvers.options['show_progress'] = False

    return_rate = []
    for col in dfr.columns:
        return_rate.append(dfr[col].values)

    n_asset    =     len(return_rate)

    asset_mean = np.mean(return_rate, axis = 1)
    #print asset_mean

    cov        =     np.cov(return_rate)

    S          =     matrix(cov)
    pbar       =     matrix(asset_mean)


    G          =     matrix(0.0, (n_asset, n_asset))
    G[::n_asset + 1]  =  -1.0
    h                 =  matrix(0.0, (n_asset, 1))
    A                 =  matrix(1.0, (1, n_asset))
    b                 =  matrix(1.0)


    N = 1000
    mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    portfolios = [ qp(mu*S, -1 * pbar, G, h, A, b)['x'] for mu in mus ]
    returns = [ dot(pbar,x) for x in portfolios ]
    risks = [ sqrt(dot(x, S*x)) for x in portfolios ]

    ws               = None
    sharpe           = -np.inf
    for i in range(0 ,len(risks)):
        if (returns[i] - 0.03 / 52) / risks[i] > sharpe:
            sharpe = (returns[i] - 0.03 / 52) / risks[i]
            ws = portfolios[i]

    weight = {}
    for i in range(0, len(dfr.columns)):
        code = dfr.columns[i]
        w    = ws[i]
        weight[code] = w
    #print weight
    return weight


def mean_cvar(dfr):

    solvers.options['show_progress'] = False

    return_rate = []
    for col in dfr.columns:
        return_rate.append(dfr[col].values)

    cvar_var   =    (1 - np.log(2 * 0.95)) / (2 ** 0.5)

    n_asset    =     len(return_rate)

    asset_mean = np.mean(return_rate, axis = 1)
    #print asset_mean

    cov        =     np.cov(return_rate)

    S          =     matrix(cov)
    pbar       =     matrix(asset_mean)


    G          =     matrix(0.0, (n_asset, n_asset))
    G[::n_asset + 1]  =  -1.0
    h                 =  matrix(0.0, (n_asset, 1))
    A                 =  matrix(1.0, (1, n_asset))
    b                 =  matrix(1.0)


    '''
    min_ws    = qp(S, 0 * pbar, G, h, A, b)['x']
    min_return= np.dot(asset_mean, min_ws)
    #print min_return
    min_risk  = sqrt(dot(min_ws, S*min_ws))
    max_risk  = max(np.std(return_rate, axis = 1))

    A                 =  matrix(1.0, (2, n_asset))
    b                 =  matrix(1.0, (2, 1))

    for i in range(0, n_asset):
        A[1, i] = asset_mean[i]
    #print np.linspace(min_return, np.max(asset_mean), 1000)
    for r in np.linspace(min_return, np.max(asset_mean), 1000):
        b[1, 0] = r
        ws      = qp(S, 0 * pbar, G, h, A, b)['x']
    '''


    N = 1000
    mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    portfolios = [ qp(mu*S, -1 * pbar, G, h, A, b)['x'] for mu in mus ]
    returns = [ dot(pbar,x) for x in portfolios ]
    risks = [ sqrt(dot(x, S*x)) for x in portfolios ]

    cvar_coefficient = (1.0 - np.log(0.1)) / (2 ** 0.5)
    cvars            = []
    ws               = None
    optimized        = np.inf
    for i in range(0 ,len(risks)):
        cvar = -1 * returns[i] + cvar_coefficient * risks[i]
        #print cvar, risks[i], returns[i]
        #if returns[i] / cvar > optimized:
        #    ws = portfolios[i]
        #    optimized = returns[i] / cvar
        if cvar < optimized:
            ws = portfolios[i]
            optimized = cvar

        #print risks[i]
    #print dfr.index
    #print dfr.columns
    #print ws
    #print risks
    #print returns
    weight = {}
    for i in range(0, len(dfr.columns)):
        code = dfr.columns[i]
        w    = ws[i]
        weight[code] = w
    return weight


if __name__ == '__main__':
    dfr = pd.read_csv('./data/fund_value2.csv', index_col = 'date', parse_dates = ['date'])
    his_back  = 36
    interval  = 12
    nav_df    = fv.FundValue(dfr, his_back, interval, mean_cvar)
    #nav_df    = fv.FundValue(dfr, his_back, interval, markowitz)
    nav_df.to_csv('nav.csv') 

