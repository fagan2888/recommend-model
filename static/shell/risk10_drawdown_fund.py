#coding=utf8


import pandas as pd
import numpy as np


def nav_max_drawdown_window(df_nav, window, min_periods=1):
    ''' calc max draw base on slice window of nav
    '''
    return df_nav.rolling(
        window=window, min_periods=min_periods).apply(
            lambda x:(x/np.maximum.accumulate(x) - 1).min())


if __name__ == '__main__':


    highrisk_df = pd.read_csv('data/nav.csv', index_col = ['date'], parse_dates = ['date'])
    #highrisk_df = highrisk_df[['risk10']]
    #highrisk_df = pd.read_csv('data/online_nav.csv', index_col = ['date'], parse_dates = ['date'])
    highrisk_df = highrisk_df[['risk10']]

    stock_fund_df = pd.read_csv('fund_value.csv', index_col = ['date'], parse_dates = ['date'])

    df = pd.concat([highrisk_df, stock_fund_df], axis = 1, join_axes = [highrisk_df.index])

    df = df.fillna(method = 'pad')

    df = df.iloc[-270 : ]

    #print df

    period = 180

    drawdown_df = nav_max_drawdown_window(df, period)
    #drawdown_df.to_csv('risk10_drawdown.csv')

    dates = df.index
    ds = []
    ranks = []
    for i in range(0, len(dates) - period):

        d = dates[i]
        next_d = dates[i + period]

        tmp_df = df.loc[d : next_d]

        drawdown = drawdown_df.loc[next_d]
        drawdown = drawdown.dropna()
        risk10_drawdown = drawdown.loc['risk10']
        #print d, len(drawdown),
        drawdown = drawdown[drawdown >= 1.2 * risk10_drawdown]
        drawdown = drawdown[drawdown <= 0.8 * risk10_drawdown]
        #print len(drawdown)
        #print d, tmp_df.columns

        tmp_df = tmp_df[drawdown.index]
        rs = tmp_df.loc[next_d] / tmp_df.loc[d] - 1
        rs = rs.dropna()
        rs.sort(ascending = False)
        index = rs.index.get_loc('risk10')
        print d, index, len(rs), 1.0 * index / len(rs)

        ds.append(d)
        ranks.append([index + 1, len(rs) + 1, 1.0 * (index + 1) / (len(rs) + 1)])
        #print rs
        #print d, df.loc[next_d, 'risk10'] / df.loc[d, 'risk10'] - 1
        #ds.append(d)
        #rs.append([df.loc[next_d, 'risk10'] / df.loc[d, 'risk10'] - 1, df.loc[next_d, '000300.SH'] / df.loc[d, '000300.SH'] - 1])

    df = pd.DataFrame(np.matrix(ranks), index = ds, columns = ['risk10_rank', 'all_fund_num', 'ratio'])
    df.to_csv('risk10_rank.csv')
    print df
    print np.mean(df['ratio'])
    #print df
