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
import scipy
import scipy.optimize

from Const import datapath

logger = logging.getLogger(__name__)

rf = 0.025 / 52

def markowitz(dfr):
    cov = dfr.cov()
    cov = cov.values
    rs = dfr.mean().values
    #print len(cov)
    asset_num = len(cov)
    w = 1.0 * np.ones(asset_num) / asset_num
    bound = [ (0.0 , 1.0) for i in range(asset_num)]
    constrain = ({'type':'eq', 'fun': lambda w: sum(w)-1.0 })
    N = 200
    ws = None
    sharpe = -1000000
    for tau in [10**(5.0*t/N-1.0) for t in range(N)]:
        result = scipy.optimize.minimize(obj_func, w, (rs, cov, tau), method='SLSQP', constraints=constrain, bounds=bound)
        tmp_ws = result.x
        returns = np.dot(tmp_ws, rs)
        risk    = np.dot( np.dot(tmp_ws, cov), tmp_ws)
        if sharpe < (returns - rf)  / (risk ** 0.5):
            sharpe = (returns - rf)  / (risk ** 0.5)
            ws = tmp_ws
    return ws


def obj_func(w, rs, cov, tau):
    val = tau * np.dot( np.dot(w, cov), w) - np.dot(w, rs)
    return val


if __name__ == '__main__':

    index = pd.read_csv('./data/index.csv', index_col = 'date', parse_dates = ['date'])
    #index = pd.read_csv('./allvdf.csv', index_col = 'date', parse_dates = ['date'])
    index = index.iloc[:,0:-1]
    #print index.columns
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
    loop_num = 52 * 2
    weight = None
    weights = []

    for i in range(look_back, len(dates)):
        d = dates[i]
        if i % interval == 0:
        #if i % 1 == 0:
            train_df_inc = df_inc.iloc[i - look_back : i]

            wss = np.zeros(len(train_df_inc.columns))
            randoms = []
            rep_num = loop_num * (look_back / 2) / look_back
            day_indexs = range(0, look_back) * rep_num
            random.shuffle(day_indexs)
            day_indexs = np.array(day_indexs)
            #print day_indexs
            day_indexs = day_indexs.reshape(len(day_indexs) / (look_back / 2), look_back / 2)
            #print day_indexs
            for m in range(0, len(day_indexs)):
                randoms.append(list(day_indexs[m]))
            #print randoms
            for j in range(0, loop_num):
                random_n = randoms[j]
                tmp_df_inc = train_df_inc.iloc[random_n]
                #tmp_df_inc = train_df_inc
                risk, returns, ws, sharpe = PF.markowitz_r(tmp_df_inc, None)
                #ws = markowitz(tmp_df_inc)
                for n in range(0, len(ws)):
                    w = ws[n]
                    wss[n] = wss[n] + w
            print d, wss / loop_num
            weight = wss / loop_num


            '''
            tmp_df_inc = train_df_inc
            #tmp_df_inc = train_df_inc
            #risk, returns, ws, sharpe = PF.markowitz_r(tmp_df_inc, None)
            ws = []
            for p in range(0, len(tmp_df_inc.columns)):
                ws.append(1.0 / len(tmp_df_inc.columns))
            weight = ws
            print d, weight
            '''

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
