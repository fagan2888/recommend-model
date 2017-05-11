#coding=utf8


import pandas as pd
import numpy as np


if __name__ == '__main__':


    highrisk_df = pd.read_csv('data/nav.csv', index_col = ['date'], parse_dates = ['date'])
    highrisk_df = highrisk_df[['risk10']]

    stock_fund_df = pd.read_csv('fund_value.csv', index_col = ['date'], parse_dates = ['date'])

    df = pd.concat([highrisk_df, stock_fund_df], axis = 1, join_axes = [highrisk_df.index])

    df = df.fillna(method = 'pad')

    #print df

    period = 1080
    dates = df.index
    ds = []
    ranks = []
    for i in range(0, len(dates) - period):

        d = dates[i]
        next_d = dates[i + period]

        tmp_df = df.loc[d : next_d]
        tmp_df = tmp_df.dropna(axis = 1)
        tmp_dfr = tmp_df.pct_change().fillna(0.0)
        #print tmp_dfr

        tmp_dfr_std = tmp_dfr.std()
        risk10_risk = tmp_dfr_std.loc['risk10']

        tmp_dfr_std = tmp_dfr_std[tmp_dfr_std >= 0.8 * risk10_risk]
        tmp_dfr_std = tmp_dfr_std[tmp_dfr_std <= 1.2 * risk10_risk]

        tmp_df = tmp_df[tmp_dfr_std.index]
        #print d, tmp_df.columns

        rs = tmp_df.loc[next_d] / tmp_df.loc[d] - 1
        rs = rs.dropna()
        rs.sort(ascending = False)
        index = rs.index.get_loc('risk10')
        print d, index, len(rs), 1.0 * index / len(rs)

        ds.append(d)
        ranks.append([index + 1, len(rs) + 1])
        #print rs
        #print d, df.loc[next_d, 'risk10'] / df.loc[d, 'risk10'] - 1
        #ds.append(d)
        #rs.append([df.loc[next_d, 'risk10'] / df.loc[d, 'risk10'] - 1, df.loc[next_d, '000300.SH'] / df.loc[d, '000300.SH'] - 1])

    df = pd.DataFrame(np.matrix(ranks), index = ds, columns = ['risk10_rank', 'all_fund_num'])
    df.to_csv('risk10_rank.csv')

    #print df
