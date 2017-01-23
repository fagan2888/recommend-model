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

from Const import datapath

logger = logging.getLogger(__name__)


if __name__ == '__main__':

    index = pd.read_csv('./data/index.csv', index_col = 'date', parse_dates = ['date'])
    index = index.fillna(method = 'pad').dropna()
    index = index.resample('W-FRI').last()
    index = index / index.iloc[0]
    df_inc = index.pct_change().fillna(0.0)
    index.to_csv('index.csv')

    ds = []
    rs = []
    dates = df_inc.index
    look_back = 52
    interval = 13
    loop_num = 100
    weight = None
    weights = []

    for i in range(look_back, len(dates)):
        d = dates[i]
        if i % interval == 0:
            train_df_inc = df_inc.iloc[i - look_back : i]
            wss = np.zeros(len(train_df_inc.columns))
            for j in range(0, loop_num):
                random_n = []
                for j in range(0, interval * 2):
                    random_n.append(random.randint(0, len(train_df_inc) - 1))
                tmp_df_inc = train_df_inc.iloc[random_n]
                #tmp_df_inc = train_df_inc
                risk, returns, ws, sharpe = PF.markowitz_r(tmp_df_inc, None)
                for n in range(0, len(ws)):
                    w = ws[n]
                    wss[n] = wss[n] + w
            print d, wss / loop_num
            weight = wss / loop_num

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

