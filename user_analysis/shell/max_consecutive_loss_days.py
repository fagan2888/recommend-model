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

    df['r'] = df[code].pct_change().fillna(0.0)

    consecutive_loss_day = {}
    loss_day = 0
    for i in range(0, len(df)):
        r = df['r'][i]
        if r < 0:
            loss_day = loss_day + 1
        else:
            num = consecutive_loss_day.setdefault(loss_day, 0)
            consecutive_loss_day[loss_day] = num + 1
            consecutive_loss_day[0] = consecutive_loss_day[0] + 1
            loss_day = 0

    print consecutive_loss_day
    num = 0
    for k,v in consecutive_loss_day.items():
        num = num + k * v
    #print num
