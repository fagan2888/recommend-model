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


def riskparity(dfr):
    cov = dfr.cov()
    cov = cov.values
    cov = cov * 10000
    #print len(cov)
    asset_num = len(cov)
    w = 1.0 * np.ones(asset_num) / asset_num
    bound = [ (0.0 , 1.0) for i in range(asset_num)]
    constrain = ({'type':'eq', 'fun': lambda w: sum(w)-1.0 })
    result = scipy.optimize.minimize(obj_func, w, (cov), method='SLSQP', constraints=constrain, bounds=bound)
    ws = result.x
    return ws


def obj_func(w, cov):
    n = len(cov)
    risk_sum = 0
    for i in range(0, n):
        for j in range(i, n):
            risk_sum = risk_sum + (np.dot(w, cov[i]) - np.dot(w , cov[j])) ** 2
    return risk_sum


if __name__ == '__main__':

    index = pd.read_csv('./data/index.csv', index_col = 'date', parse_dates = ['date'])
    #index = index.iloc[:,0:-1]
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
    loop_num = 52 * 4
    weight = None
    weights = []

    for i in range(look_back, len(dates)):
        d = dates[i]
        #if i % interval == 0:
        if i % 1 == 0:
            train_df_inc = df_inc.iloc[i - look_back : i].copy()
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
                ws = riskparity(tmp_df_inc)
                for n in range(0, len(ws)):
                    w = ws[n]
                    wss[n] = wss[n] + w

            print d, wss / loop_num
            weight = wss / loop_num
            #train_df_inc.iloc[:,-1] = train_df_inc.iloc[:,-1] * 16


            '''
            tmp_df_inc = train_df_inc
            #tmp_df_inc = train_df_inc
            risk, returns, ws, sharpe = PF.markowitz_r(tmp_df_inc, None)
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
    vdf.to_csv('riskparitynav.csv')


    pdf = pd.DataFrame(weights, index = ds, columns = index.columns)
    pdf.index.name = 'date'
    pdf.to_csv('riskparityposition.csv')
    presult = pdf.rolling(window = 2, min_periods = 1).apply(lambda x : x[1] - x[0] if len(x) > 1 else x[0])
    presult = presult.abs().sum(axis = 1).to_frame('turnover').sum()
    print presult

