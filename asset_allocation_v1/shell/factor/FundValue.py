#coding=utf8


import numpy  as np
import pandas as pd
import time


def equal_weight(dfr):
    ws = {}
    codes = dfr.columns
    for i in range(0, len(codes)):
        code = codes[i]
        ws[code] = (1.0 / len(codes))
    return ws


def FundValue(dfr, his_back, interval, allocation):

    vs       = []
    ds       = []
    dates    = dfr.index
    ws       = []
    codes    = []

    for i in range(his_back, len(dates)):

        d = dates[i]
        if (i - his_back) % interval == 0:
            tmp_dfr  = dfr.iloc[i - his_back : i, ]
            ws       = allocation(tmp_dfr)
            codes    = list(ws.keys())

        r = 0
        for j in range(0, len(codes)):
            code = codes[j]
            r = r + dfr.loc[d, code] * ws[code]

        if len(vs) == 0:
            vs.append(1.0)
        else:
            v = vs[-1] * (1 + r)
            vs.append(v)

        ds.append(d)
        print d, vs[-1]


    result_df = pd.DataFrame(vs, index = ds, columns = ['nav'])
    return result_df


if __name__ == '__main__':

    fdf = pd.read_csv('./data/fund_value.csv', index_col = 'date', parse_dates = ['date'])
    fdf = fdf.iloc[-989:-1,]
    fdf.dropna(axis = 1, inplace = True)
    fdf = fdf.resample('W-FRI').last()
    fdfr = fdf.pct_change().fillna(0.0)

    his_back = 52
    interval = 1
    vdf = FundValue(fdfr, his_back, interval ,equal_weight)
