#coding=utf8

import pandas as pd
import numpy as np


if __name__ == '__main__':


    highrisk_df = pd.read_csv('data/nav.csv', index_col = ['date'], parse_dates = ['date'])

    hs300_df = pd.read_csv('data/000300.csv', index_col = ['date'], parse_dates = ['date'])

    #print highrisk_df

    df = pd.concat([highrisk_df, hs300_df], axis = 1, join_axes = [highrisk_df.index])

    df = df / df.iloc[0]

    df = df.fillna(method = 'pad')

    dates = df.index
    ds = []
    rs = []
    for i in range(0, len(dates) - 90):
        d = dates[i]
        next_d = dates[i + 90]

        #print d, df.loc[next_d, 'risk10'] / df.loc[d, 'risk10'] - 1
        ds.append(d)
        rs.append([df.loc[next_d, 'risk10'] / df.loc[d, 'risk10'] - 1, df.loc[next_d, '000300.SH'] / df.loc[d, '000300.SH'] - 1])

    df = pd.DataFrame(np.matrix(rs), index = ds, columns = ['risk10', '000300.SH'])

    print df
    df.to_csv('risk10_000300.csv')
