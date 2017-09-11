#coding=utf8

import numpy as np
import pandas as pd


if __name__ == '__main__':


    a_month_day = 5
    #code = 'nav'
    code = '000001.SH'
    df = pd.read_csv('./data/nav.csv', index_col = ['date'])
    df = df[[code]]
    #print df / df.iloc[0]
    df['cummax'] = df[code].cummax()
    df['drawdown'] = 1.0 - df[code] / df['cummax']
    df_inc = df
    df_inc[code] = df_inc[code].pct_change().fillna(0.0)

    #df_inc = df.pct_change().fillna(0.0)
    df_inc['a_month_day'] = df_inc[code].rolling(window = a_month_day).sum()
    df_inc['a_month_day'] = df_inc['a_month_day'].shift(-1 * a_month_day)
    df_inc = df_inc.dropna()

    dates = df_inc.index
    all_r = []
    for i in range(0, len(dates)):
        rs = df_inc[code][i - 0 : i ]
        #if rs[0] < 0 and rs[1] < 0 and rs[2] < 0 and rs[3] < 0 and rs[4] < 0:
        #if rs[0] < 0 and rs[1] < 0 and rs[2] < 0:
        if df.loc[dates[i] , 'drawdown'] > 0.01:
            r = df_inc['a_month_day'][i]
            all_r.append(r)
    rs = np.array(all_r)
    print rs.mean()
    print 1.0 * (rs > 0).sum() / len(rs)
    print len(rs)
