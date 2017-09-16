#coding=utf8


import numpy as np
import pandas as pd


if __name__ == '__main__':

    #code = 'nav'
    code = '000001.SH'
    df = pd.read_csv('./data/nav.csv', index_col = ['date'], parse_dates = ['date'])
    df = df / df.iloc[0]
    df = df[[code]]
    df = df.resample('M').last()

    share = 100000 / df[code][0]
    principal = 100000
    money = [100000]
    add_money = 130000.0 / len(df)
    for i in range(0, len(df)):
        v = df[code][i]
        share = share + add_money / v
        money.append(share * v)
    #print share * df[code][-1]
    print money
    #print principal, money[-1]
