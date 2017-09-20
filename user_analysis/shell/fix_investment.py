#coding=utf8


import numpy as np
import pandas as pd


if __name__ == '__main__':


    code = 'nav'
    #code = '000001.SH'
    df = pd.read_csv('./data/nav.csv', index_col = ['date'])
    df = df[df.index >= '2015-08-01']
    df = df[df.index <= '2017-08-31']
    df = df / df.iloc[0]
    df = df[[code]]

    #df['cummax'] = df[code].cummax()
    #df['drawdown'] = 1.0 - df[code] / df['cummax']
    df['r'] = df[code].pct_change().fillna(0.0)

    share = 100000 / df[code][0]
    principal = 100000
    money = [100000]
    loss_num = 0
    drawdown_threshold = 0.05
    ds = []
    m = 200000 / 24.0
    for i in range(0, len(df) - 1):
        v = df[code][i + 1]
        if i % 21 == 0:
            share = share + m / v
            principal = principal + m
        money.append(v * share)
    #print len(money)
    #print len(df.index)
    money_df = pd.DataFrame(money, index = df.index)
    print principal, money[-1]
    #print money_df
    money_df.to_csv('money.csv')
