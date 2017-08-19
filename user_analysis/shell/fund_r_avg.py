#coding=utf8


import pandas as pd
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('./data/fund_nav.csv', parse_dates = ['date'], index_col = ['date'])
    df = df.fillna(method = 'pad')
    dfr = df.pct_change()
    dates = df.index

    rs = []
    ds = []
    for d in dates:
        r = dfr.loc[d]
        r = np.mean(r)
        ds.append(d)
        rs.append(r)

    df = pd.DataFrame(rs, index = ds, columns = ['r'])

    df.to_csv('fund_inc_avg.csv')
