#coding=utf8


import pandas as pd
import numpy as np


def cppi(df, high_risk_level, low_risk_level, max_loss, multi_v):

    low_df = df[low_risk_level]
    high_df = df[high_risk_level]

    low_dfr = low_df.pct_change().fillna(0.0)
    high_dfr = high_df.pct_change().fillna(0.0)

    low_average_return = np.percentile(low_dfr.values, 50)
    #high_great_loss = np.percentile(high_dfr.values, 5)

    dates = df.index

    vs = []
    ds = []
    high_pos = []

    for i in range(0, len(dates)):

        d = dates[i]

        if len(vs) == 0:
            v = 1.0
        else:
            v = vs[-1]

        high_v = (v * (1 + low_average_return) - 1 * (1 - max_loss)) * multi_v

        if high_v > v:
            high_v = v
        elif high_v < 0:
            high_v = 0
        low_v = v - high_v

        high_r = high_dfr.loc[d]
        low_r = low_dfr.loc[d]

        now_v = high_v * (1 + high_r) + low_v * (1 + low_r)
        high_pos.append(high_v / v)
        vs.append(now_v)
        ds.append(d)
        print d, high_v / v, now_v


    df = pd.DataFrame(np.matrix([high_pos, vs]).T, index = ds, columns = ['pos', 'nav'])
    df.index.name = 'date'
    #print df
    return df
    #print low_average_return, high_great_loss


if __name__ == '__main__':


    df = pd.read_csv('./data/nav.csv', index_col = ['date'], parse_dates = ['date'])
    #df = df.iloc[-400:]
    df = df.resample('W-FRI').last()

    result_df = cppi(df, 'risk10', 'risk1', 0.01, 10)
    result_df = result_df[['pos']]
    print result_df
