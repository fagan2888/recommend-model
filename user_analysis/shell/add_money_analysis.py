#coding=utf8


import numpy as np
import pandas as pd


if __name__ == '__main__':


    a_month_day = 22
    #code = 'nav'
    code = '000001.SH'
    df = pd.read_csv('./data/nav.csv', index_col = ['date'])
    df = df[[code]]
    df = df[df.index >= '2015-08-01']
    df = df[df.index <= '2017-08-31']
    #print df / df.iloc[0]
    df['cummax'] = df[code].cummax()
    df['drawdown'] = 1.0 - df[code] / df['cummax']
    df_inc = df
    df_inc[code] = df_inc[code].pct_change().fillna(0.0)

    #df_inc = df.pct_change().fillna(0.0)
    df_inc['a_month_day'] = df_inc[code].rolling(window = a_month_day).sum()
    df_inc['a_month_day'] = df_inc['a_month_day'].shift(-1 * a_month_day)
    df_inc = df_inc.dropna()
    df_inc.to_csv('inc.csv')

    dates = df_inc.index
    all_r = []
    loss_num = 0
    drawdown = []
    for i in range(0, len(dates)):
        r = df_inc[code][i]
        if r < 0:
            loss_num = loss_num + 1
        if loss_num == 5:
            loss_num = 0
            drawdown.append( np.sum(df_inc[code][i - 5: i]) )
            all_r.append( df_inc['a_month_day'][i] )
            print df_inc.index[i]
        if r >= 0:
            loss_num = 0

    #print drawdown
    #print all_r
    print np.mean(drawdown)
    for i in range(0, len(drawdown)):
        print all_r[i], drawdown[i]
