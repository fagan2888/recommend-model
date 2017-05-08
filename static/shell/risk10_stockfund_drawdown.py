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
    highrisk_df = highrisk_df[['risk10']]

    stock_fund_df = pd.read_csv('stock_fund_value.csv', index_col = ['date'], parse_dates = ['date'])

    df = pd.concat([highrisk_df, stock_fund_df], axis = 1, join_axes = [highrisk_df.index])

    df = df.fillna(method = 'pad')

    #print df

    window = 1080
    drawdown_df = nav_max_drawdown_window(df, window)
    drawdown_df.to_csv('risk10_drawdown.csv')

    dates = drawdown_df.index
    drawdown_rank = []
    ds = []
    for i in range(window, len(dates)):

        d = dates[i]
        ds.append(d)

        drawdown = drawdown_df.iloc[i]
        drawdown = drawdown.dropna()
        drawdown.sort(ascending=False)
        #print drawdown
        index = drawdown.index.get_loc('risk10')
        print dates[i], index, len(drawdown)
        drawdown_rank.append([index, len(drawdown), 1.0 * index / len(drawdown)])


    drawdown_rank_df = pd.DataFrame(np.matrix(drawdown_rank), index = ds,  columns = ['risk10', 'stock_fund', 'ratio'])
    #print drawdown_rank_df
    print np.mean(drawdown_rank_df['ratio'])
