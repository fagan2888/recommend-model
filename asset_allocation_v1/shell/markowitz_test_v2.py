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
import multiprocessing


from Const import datapath

logger = logging.getLogger(__name__)

rf = 0.025 / 52

def riskparity(dfr):
    cov = dfr.cov()
    cov = cov.values
    cov = cov * 10000
    #print len(cov)
    asset_num = len(cov)
    w = 1.0 * np.ones(asset_num) / asset_num
    bound = [ (0.0 , 1.0) for i in range(asset_num)]
    constrain = ({'type':'eq', 'fun': lambda w: sum(w)-1.0 })
    result = scipy.optimize.minimize(riskparity_obj_func, w, (cov), method='SLSQP', constraints=constrain, bounds=bound)
    ws = result.x
    return ws


def riskparity_obj_func(w, cov):
    n = len(cov)
    risk_sum = 0
    for i in range(0, n):
        for j in range(i, n):
            risk_sum = risk_sum + (np.dot(w, cov[i]) - np.dot(w , cov[j])) ** 2
    return risk_sum


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
        result = scipy.optimize.minimize(markowitz_obj_func, w, (rs, cov, tau), method='SLSQP', constraints=constrain, bounds=bound)
        tmp_ws = result.x
        returns = np.dot(tmp_ws, rs)
        risk    = np.dot( np.dot(tmp_ws, cov), tmp_ws)
        if sharpe < (returns - rf)  / (risk ** 0.5):
            sharpe = (returns - rf)  / (risk ** 0.5)
            ws = tmp_ws
    return ws


def markowitz_obj_func(w, rs, cov, tau):
    val = tau * np.dot( np.dot(w, cov), w) - np.dot(w, rs)
    return val


def m_markowitz(queue, random_index, df_inc, bound):
    for index in random_index:
        tmp_df_inc = df_inc.iloc[index]
        #ws = riskparity(tmp_df_inc)
        risk, returns, ws, sharpe = PF.markowitz_r_spe(tmp_df_inc, bound)
        queue.put(ws)


if __name__ == '__main__':


    index = pd.read_csv('./data/index.csv', index_col = 'date', parse_dates = ['date'])
    #index = pd.read_csv('./allvdf.csv', index_col = 'date', parse_dates = ['date'])
    #index = index.iloc[:,0:-1]
    #print index.columns
    index = index.fillna(method = 'pad').dropna()
    index = index.resample('W-FRI').last()
    index = index / index.iloc[0]
    index.to_csv('index.csv')

    #index = index[index.columns[2:]]
    #print index.columns
    df_inc = index.pct_change().fillna(0.0)


    bound_set = {
        '000300.SH': {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        '000905.SH': {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'SPX.GI':     {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'NDX.GI':  {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'HSCI.HI':   {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'SPTAUUSDOZ.IDC':    {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'B.IPE':    {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
    }

    bound = []
    for asset in df_inc.columns:
        bound.append(bound_set[asset])


    ds = []
    rs = []
    dates = df_inc.index
    look_back = 52
    interval = 13
    loop_num = 52 * 4
    weight = None
    weights = []

    for i in range(look_back, len(dates)):
        d = dates[i]
        #if i % interval == 0:
        if i % 1 == 0:
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

            q = multiprocessing.Queue()
            process_num = 4
            processes = []
            process_index_num = len(randoms) / process_num
            for m in range(0, process_num):
                indexs = randoms[m * process_index_num : (m + 1) * process_index_num]
                p = multiprocessing.Process(target = m_markowitz, args = (q, indexs, train_df_inc, bound,))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            for m in range(0, q.qsize()):
                ws = q.get(m)
                for n in range(0, len(ws)):
                    w = ws[n]
                    wss[n] = wss[n] + w

            weight = wss / loop_num
            print d, weight

            '''
            tmp_df_inc = train_df_inc
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
    vdf = pd.concat([index, vdf], axis = 1, join_axes = [vdf.index])
    vdf = vdf / vdf.iloc[0]
    vdf.to_csv('robustmarkowitznav.csv')


    pdf = pd.DataFrame(weights, index = ds, columns = index.columns)
    pdf.index.name = 'date'
    pdf.to_csv('robustmarkowitzposition.csv')
    presult = pdf.rolling(window = 2, min_periods = 1).apply(lambda x : x[1] - x[0] if len(x) > 1 else x[0])
    presult = presult.abs().sum(axis = 1).to_frame('turnover').sum()
    print presult
