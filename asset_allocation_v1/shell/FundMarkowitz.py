#coding=utf8


import sys
sys.path.append('shell')
import pandas as pd
import Portfolio
import numpy as np


if __name__ == '__main__':


    df = pd.read_csv('money_fund.csv', parse_dates = ['date'], index_col = ['date'])
    df = df.resample('W-FRI').last()
    df = df.fillna(method = 'pad')
    dfr = df.pct_change().fillna(0.0)

    pos_df = pd.read_csv('money_fund_pos.csv', parse_dates = ['date'], index_col = ['date'])

    dfr = dfr.loc[pos_df.index]

    rs = []
    for i in range(0, 18, 3):
        pos = pos_df.iloc[:,i:i + 3]
        pos.columns = pos_df.iloc[:,0 : 3].columns
        r = dfr * pos
        r = r.sum(axis = 1)
        rs.append(r)

    df = pd.concat(rs, axis = 1, join_axes = [pos_df.index])
    df = (df + 1).cumprod()
    df.to_csv('nav.csv')

    '''
    ds = []
    ws = []
    for i in range(26, len(dfr)):

        tmp_dfr = dfr.iloc[i - 26 : i]
        bound = []
        for col in dfr.columns:
            bound.append({'sum1': 0,    'sum2' : 0,   'upper': 1.0,  'lower': 0.0})

        risk, ret, wss, sharpe = Portfolio.markowitz_bootstrape(tmp_dfr, bound)
        #tmp_sum = tmp_dfr.sum()
        #wss = []
        #for i in range(0, len(tmp_sum)):
        #    if tmp_sum.iloc[i] == min(tmp_sum):
        #        wss.append(1.0)
        #    else:
        #        wss.append(0.0)
        ws.append(wss)
        ds.append(dfr.index[i])
        print dfr.index[i]

    pos = pd.DataFrame(ws, index = ds, columns = dfr.columns)
    pos = pos.rolling(window = 4, min_periods = 1).mean()
    #print pos
    dfr = dfr.loc[pos.index]

    nav = pos * dfr
    nav = nav.sum(axis = 1)
    nav = (nav + 1).cumprod()
    print nav
    nav.to_csv('nav.csv')
    pos.to_csv('pos.csv')
    '''


