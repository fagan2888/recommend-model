#coding=utf8


import pandas as pd
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('./data/nav.csv', index_col = ['date'], parse_dates = ['date'])
    df = df.iloc[-400:]
    df = df.resample('W-FRI').last()

    low_df = df['risk1']
    high_df = df['risk10']

    low_dfr = low_df.pct_change().fillna(0.0)
    high_dfr = high_df.pct_change().fillna(0.0)

    low_return = np.percentile(low_dfr.values, 50)
    #high_drawdown = np.percentile(high_dfr.values, 5)

    #print low_return, high_drawdown
    #print low_return, high_drawdown

    dates = low_df.index
    high_dfr = high_df.pct_change().fillna(0.0)
    low_dfr = low_df.pct_change().fillna(0.0)


    multi_v = 15
    vs = []
    highw = 0
    loww = 0
    ds = []
    pos = []
    for i in range(0, len(dates)):
        d = dates[i]
        if len(vs) == 0:
            v = 1.0
        else:
            v = vs[-1]

        high = (v * (1 + low_return) - 1) * multi_v
        if high > v:
            high = v
        elif high < 0:
            high = 0
        low = v - high

        highr = high_dfr.loc[d]
        lowr = low_dfr.loc[d]

        now_v = high * (1 + highr) + low * (1 + lowr)
        pos.append(high / v)
        vs.append(now_v)
        ds.append(d)
        print d, high / v, now_v

    df = pd.DataFrame(np.matrix([pos, vs]).T, index = ds, columns = ['pos', 'nav'])
    df.to_csv('cppi.csv')


    #print np.mean(three_month_return)
    #print three_month_return


    #low_df =
    #print low_df
    #print high_df
