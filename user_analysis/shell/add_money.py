#coding=utf8


import numpy as np
import pandas as pd


if __name__ == '__main__':

    code = 'nav'
    #code = '000001.SH'
    df = pd.read_csv('./data/nav.csv', index_col = ['date'])
    df = df / df.iloc[0]
    df = df[[code]]

    df['cummax'] = df[code].cummax()
    df['drawdown'] = 1.0 - df[code] / df['cummax']
    df['r'] = df[code].pct_change().fillna(0.0)

    share = 100000 / df[code][0]
    principal = 100000
    money = [100000]
    loss_num = 0
    drawdown_threshold = 0.05
    for i in range(0, len(df) - 1):

        v = df[code][i + 1]
        drawdown = df['drawdown'][i]
        if drawdown >= drawdown_threshold:
            share = share + 10000 / v
            principal = principal + 10000
            drawdown_threshold = drawdown_threshold + 0.05
        elif drawdown == 0.0:
            drawdown_threshold = 0.05
        money.append(v * share)
        #tmp_df = df.iloc[i : ]
        #tmp_df['cummax'] = tmp_df[code].cummax()
        #tmp_df['drawdown'] = 1.0 - tmp_df[code] / tmp_df['cummax']


        '''
        r = df['r'][i]
        v = df[code][i + 1]
        if r < 0:
            loss_num = loss_num + 1
        else:
            loss_num = 0
        if loss_num >= 5:
            loss_num = 0
            share = share + 10000 / v
            principal = principal + 10000
        money.append(v * share)
        '''
    print principal, money[-1]
