#coding=utf8


import pandas as pd
import numpy as np
import scipy.optimize



def obj_fun(x, target_r, n):
    p = 1
    for i in range(0, n):
       p = p * ((((1 - 1.0 * i / n) ** 0.5) * x) + 1 )
    return (p - target_r) ** 2


def target_life_risklevels(df, target_r, n, interval):
    df = df.iloc[::interval]
    dfr = df.pct_change().fillna(0.0)
    rs_dfr = dfr.mean()
    rs = list(rs_dfr.values)
    rs.sort()

    target_rs = []
    x = 0
    res = scipy.optimize.minimize(obj_fun, x, (target_r, n), method='SLSQP')
    x = res.x[0]
    for i in range(0, n):
        target_rs.append(((1 - 1.0 * i / n) ** 0.5) * x)
    target_rs.sort(reverse = True)


    risk_levels = []
    for i in range(0, len(target_rs)):
        target_r = target_rs[i]
        for j in range(0, len(rs)):
            if (j == 0 and target_r  < rs[j]) or (target_r >= rs[j - 1] and target_r <= rs[j]):
                #print j + 1
                risk_levels.append(j + 1)
                break

    return risk_levels


def prob(df, target_r, risk_levels, n, interval):

    df = df.iloc[::interval]
    dfr = df.pct_change().fillna(0.0)

    total = 0
    num = 0
    dates = dfr.index
    for i in range(0, len(dates) - n):
        v = 1
        for j in range(0, len(risk_levels)):
            risk_level = risk_levels[j]
            d = dates[ i + j ]
            r = dfr.loc[d, 'risk' + str(risk_level)]
            v = v * (r + 1)
        print dates[i], v
        total = total + 1
        if v > 1.06:
            num = num + 1

    print total
    print num
    print 1.0 * num / total
    #print rs
    #print target_rs

if __name__ == '__main__':


    df = pd.read_csv('./data/nav.csv', index_col = ['date'], parse_dates = ['date'])
    df = df.resample('M', how='last')

    target_r = 1.10
    n = 12
    interval = 1

    risk_levels = target_life_risklevels(df, target_r, n, interval)
    risk_levels = []
    for i in range(0, n):
        risk_levels.append(10)
    prob(df, target_r, risk_levels, n , interval)
    print risk_levels
