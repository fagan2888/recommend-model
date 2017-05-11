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
    #highrisk_df = pd.read_csv('data/online_nav.csv', index_col = ['date'], parse_dates = ['date'])
    highrisk_df = highrisk_df[['risk10']]

    stock_fund_df = pd.read_csv('stock_fund_value.csv', index_col = ['date'], parse_dates = ['date'])

    df = pd.concat([highrisk_df, stock_fund_df], axis = 1, join_axes = [highrisk_df.index])

    df = df.fillna(method = 'pad')

    df = df.iloc[-270 : ]
    print df
    #print df

    window = 90
    drawdown_df = nav_max_drawdown_window(df, window)
    drawdown_df.to_csv('risk10_drawdown.csv')

    dfr = df.rolling(window = window, min_periods = 1).apply(lambda x : x[-1] / x[0] - 1)

    dates = drawdown_df.index
    drawdown_rank = []
    ds = []
    for i in range(window, len(dates)):

        d = dates[i]

        rs = dfr.loc[d]
        rs = rs.dropna()
        risk10_r = rs['risk10']
        sign = np.sign(risk10_r)
        if sign > 0:
            rs = rs[rs >= 0.8 * risk10_r]
            rs = rs[rs <= 1.2 * risk10_r]
        else:
            rs = rs[rs <= 0.8 * risk10_r]
            rs = rs[rs >= 1.2 * risk10_r]

        #print rs.index
        drawdown = drawdown_df[rs.index]
        drawdown = drawdown.iloc[i]
        drawdown = drawdown.dropna()
        drawdown.sort(ascending=False)
        #print drawdown
        index = drawdown.index.get_loc('risk10')
        print dates[i], index, len(drawdown)
        if len(drawdown) <= 0:
            continue
        drawdown_rank.append([index + 1, len(drawdown) + 1, 1.0 * index / len(drawdown)])
        ds.append(d)


    drawdown_rank_df = pd.DataFrame(np.matrix(drawdown_rank), index = ds,  columns = ['risk10', 'stock_fund', 'ratio'])
    #print drawdown_rank_df
    print np.mean(drawdown_rank_df['ratio'])
