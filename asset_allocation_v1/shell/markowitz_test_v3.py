#coding=utf8


import os
import string
import click
import logging
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import Portfolio as PF
import DBData
import DFUtil
import AllocationData
import random
import cvxopt as opt
from cvxopt import blas, solvers, matrix
from cvxopt.solvers import *
from cvxopt.blas import dot
from math import sqrt
import sklearn

from Const import datapath

logger = logging.getLogger(__name__)


def markowitz(funddfr):

        return_rate = []

        for code in funddfr.columns:
            return_rate.append(funddfr[code].values)


        solvers.options['show_progress'] = False

        n_asset    =     len(return_rate)

        asset_mean = np.mean(return_rate, axis = 1)
    #print asset_mean

        l = len(asset_mean)
        cov        =     np.cov(return_rate)
        print cov
        cov = sklearn.covariance.OAS().fit(cov)
        print cov
        S          =     matrix(cov)
        l2 = matrix(S * np.eye(l))
        l2 = l2 * 2
        S = S + l2
        pbar       =     matrix(asset_mean)
        G          =     matrix(0.0, (n_asset, n_asset))
        G[::n_asset + 1]  =  -1.0
        h                 =  matrix(0.0, (n_asset, 1))
        A                 =  matrix(1.0, (1, n_asset))
        b                 =  matrix(1.0)

        N = 200
        mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
        portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
        returns = [ dot(pbar,x) for x in portfolios ]
        risks = [ sqrt(dot(x, S*x)) for x in portfolios ]

        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        #x1 = m1[2] / m1[0]
        # CALCULATE THE OPTIMAL PORTFOLIO
        #print m1
        #print x1
        wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
        return np.asarray(wt)
        #return risks, returns, portfolios


if __name__ == '__main__':

    index = pd.read_csv('./data/index.csv', index_col = 'date', parse_dates = ['date'])
    index = index.fillna(method = 'pad').dropna()
    index = index.resample('W-FRI').last()
    index = index / index.iloc[0]
    index.to_csv('index.csv')

    #index = index[index.columns[2:]]
    #print index.columns
    df_inc = index.pct_change().fillna(0.0)

    ds = []
    rs = []
    dates = df_inc.index
    look_back = 52
    interval = 13
    loop_num = 52
    weight = None
    weights = []

    for i in range(look_back, len(dates)):
        d = dates[i]
        if i % interval == 0:
            train_df_inc = df_inc.iloc[i - look_back : i]

            '''
            wss = np.zeros(len(train_df_inc.columns))
            freq_dict = {}
            randoms = []
            for j in range(0, loop_num):
                random_n = []
                m = 0
                while m < interval * 2 * loop_num / look_back:
                    rint = random.randint(0, len(train_df_inc) - 1)
                    rfreq = freq_dict.setdefault(rint, 0)
                    if rfreq < interval * 2:
                        random_n.append(rint)
                        freq_dict[rint] = rfreq + 1
                        m += 1
                randoms.append(random_n)

            for j in range(0, loop_num):
                random_n = randoms[j]
                tmp_df_inc = train_df_inc.iloc[random_n]
                #tmp_df_inc = train_df_inc
                risk, returns, ws, sharpe = PF.markowitz_r(tmp_df_inc, None)
                for n in range(0, len(ws)):
                    w = ws[n]
                    wss[n] = wss[n] + w
            print d, wss / loop_num
            weight = wss / loop_num
            '''

            tmp_df_inc = train_df_inc
            tmp_df_inc = tmp_df_inc.iloc[:,0:2]
            #tmp_df_inc = train_df_inc
            risk, returns, ws, sharpe = PF.markowitz_r(tmp_df_inc, None)
            weight = np.asarray(ws)
            #print d, weight
            wt = markowitz(tmp_df_inc)
            #print d, wt
            weight = wt


        r = 0
        for n in range(0, len(weight)):
            r = r + df_inc.iloc[i, n] * weight[n]
        ds.append(d)
        rs.append(r)
        weights.append(weight)

    vdf = pd.DataFrame(rs, index = ds, columns = ['nav'])
    vdf.index.name = 'date'
    vdf = (1 + vdf).cumprod()
    vdf.to_csv('robustmarkowitznav.csv')


    pdf = pd.DataFrame(weights, index = ds, columns = index.columns)
    pdf.index.name = 'date'
    pdf.to_csv('robustmarkowitzposition.csv')
    presult = pdf.rolling(window = 2, min_periods = 1).apply(lambda x : x[1] - x[0] if len(x) > 1 else x[0])
    presult = presult.abs().sum(axis = 1).to_frame('turnover').sum()
    print presult

