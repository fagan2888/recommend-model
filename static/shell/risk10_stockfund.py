#coding=utf8


import pandas as pd
import numpy as np


if __name__ == '__main__':


    highrisk_df = pd.read_csv('data/nav.csv', index_col = ['date'], parse_dates = ['date'])
    highrisk_df = highrisk_df[['risk10']]

    stock_fund_df = pd.read_csv('stock_fund_value.csv', index_col = ['date'], parse_dates = ['date'])

    df = pd.concat([highrisk_df, stock_fund_df], axis = 1, join_axes = [highrisk_df.index])

    df = df.fillna(method = 'pad')

    #print df

    dates = df.index
    ds = []
    ranks = []
    for i in range(0, len(dates) - 1080):
        d = dates[i]
        next_d = dates[i + 1080]

        rs = df.loc[next_d] / df.loc[d] - 1
        rs = rs.dropna()
        rs.sort(ascending = False)
        index = rs.index.get_loc('risk10')
        print d, index, len(rs), 1.0 * index / len(rs)

        ds.append(d)
        ranks.append([index, len(rs)])
        #print rs
        #print d, df.loc[next_d, 'risk10'] / df.loc[d, 'risk10'] - 1
        #ds.append(d)
        #rs.append([df.loc[next_d, 'risk10'] / df.loc[d, 'risk10'] - 1, df.loc[next_d, '000300.SH'] / df.loc[d, '000300.SH'] - 1])

    df = pd.DataFrame(np.matrix(ranks), index = ds, columns = ['risk10_rank', 'all_fund_num'])
    df.to_csv('risk10_rank.csv')

    #print df
