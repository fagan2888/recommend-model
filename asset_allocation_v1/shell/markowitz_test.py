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

    funddf = pd.read_csv('./tmp/equalriskasset.csv', index_col = 'date', parse_dates = ['date'])
    funddf = funddf.resample('W-FRI').last()
    funddf = funddf.fillna(method = 'pad').dropna()
    fundcols = ['largecap', 'smallcap', 'GLNC', 'HSCI.HI', 'SP500.SPI']
    funddf = funddf[fundcols]
    funddfr = funddf.pct_change().fillna(0.0)

    index = DBData.db_index_value_daily('2010-01-01', '2017-01-13', None)
    index = index.resample('W-FRI').last()
    index = index.fillna(method = 'pad').dropna()
    index = index / index.iloc[0]
    #print index.columns
    cols = ['000300.SH', '000905.SH', 'GLNC', 'HSCI.HI', 'SP500.SPI']
    #cols = ['000300.SH', '000905.SH', 'GLNC', 'HSCI.HI', 'SP500.SPI', 'H11001.CSI']
    index = index[cols]
    index.to_csv('index.csv')
    df_inc = index.pct_change().fillna(0.0)

    '''
    df_inc = df_inc.iloc[-52:]
    while True:
        ratio = 1.0 * random.randint(1, 99) / 100
        wss = np.zeros(len(cols))
        loop_num = random.randint(50 , 150)
        for i in range(0, loop_num):
            random_n = []
            for j in range(0, int(len(df_inc) * ratio)):
                random_n.append(random.randint(0, len(df_inc) - 1))
            tmp_df_inc = df_inc.iloc[random_n]
            risk, returns, ws, sharpe = PF.markowitz_r(tmp_df_inc, None)
            #print ws
            for n in range(0, len(ws)):
                w = ws[n]
                wss[n] = wss[n] + w
        print ratio, loop_num, wss / loop_num
    #print np.mean(wss)
    '''

    '''
    alldf = []
    allrs = []

    for m in range(0, 10):
        ds = []
        rs = []

        dates = df_inc.index
        look_back = 52
        interval = 13
        loop_num = 100
        weight = None

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

        allrs.append(rs)

    df = pd.DataFrame(np.matrix(allrs).T, index = ds)
    df.index.name = 'date'
    df = (1 + df).cumprod()
    df.to_csv('nav.csv')

    '''

    ds = []
    rs = []
    dates = df_inc.index
    look_back = 52
    interval = 13
    loop_num = 100
    weight = None

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

        rindex = 0
        rfund = 0
        for n in range(0, len(weight)):
            rindex = rindex + df_inc.iloc[i, n] * weight[n]
            rfund = rfund + funddfr.loc[d, fundcols[n]] * weight[n]
            #r = r + df_inc.iloc[i, n] * 1.0 / len(weight)
        ds.append(d)
        rs.append([rindex, rfund])

    df = pd.DataFrame(rs, index = ds)
    df.index.name = 'date'
    df = (1 + df).cumprod()
    df.to_csv('nav.csv')
