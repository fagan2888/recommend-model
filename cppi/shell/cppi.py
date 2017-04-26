#coding=utf8


import pandas as pd
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('./data/nav.csv', index_col = ['date'], parse_dates = ['date'])

    df = df.resample('W-FRI').last()

    low_df = df['risk1']
    high_df = df['risk10']

    low_dfr = low_df.pct_change().fillna(0.0)
    high_dfr = high_df.pct_change().fillna(0.0)

    np.percentile(low_dfr.values, 50)
    np.percentile(high_dfr.values, 5)


    '''
    three_month_return = []
    dates = low_df.index
    for i in range(0, len(dates) - 3):
        startv = low_df.loc[dates[i]]
        endv = low_df.loc[dates[i + 3]]
        three_month_return.append( endv / startv - 1)
    low_return = np.percentile(three_month_return, 50)


    three_month_return = []
    dates = high_df.index
    for i in range(0, len(dates) - 3):
        startv = high_df.loc[dates[i]]
        endv = high_df.loc[dates[i + 3]]
        three_month_return.append( endv / startv - 1)
    high_drawdown = np.percentile(three_month_return, 5)
    '''

    #print low_return, high_drawdown


    low_df = low_df.iloc[::3]
    high_df = high_df.iloc[::3]

    dates = low_df.index
    high_dfr = high_df.pct_change().fillna(0.0)
    low_dfr = low_df.pct_change().fillna(0.0)


    multi_v = 15
    vs = []
    highw = 0
    loww = 0
    ds = []
    for i in range(0, len(dates)):
        d = dates[i]
        if len(vs) == 0:
            v = 1.0
        else:
            v = vs[-1]

        high = (v * low_return) * multi_v
        low = v - high

        highr = high_dfr.loc[d]
        lowr = low_dfr.loc[d]

        vs.append(high * (1 + highr) + low * (1 + lowr))
        ds.append(d)

    df = pd.DataFrame(vs, index = ds, columns = ['nav'])
    df.to_csv('cppi.csv')
    #print np.mean(three_month_return)
    #print three_month_return


    #low_df =
    #print low_df
    #print high_df
