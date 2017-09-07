#coding=utf8


#from keras.preprocessing import sequence
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.layers import Embedding
#from keras.layers import LSTM
#from keras.layers import Conv1D, MaxPooling1D


import pandas as pd
import numpy as np


if __name__ == '__main__':


    df = pd.read_csv('./data/interest.csv', index_col = ['date'], parse_dates = ['date'])
    dates = pd.date_range(df.index[0], df.index[-1])
    df = df.reindex(dates)
    df = df.fillna(method = 'pad')
    df = df.loc[dates]

    dfr = df.pct_change()

    length = 360

    columns = ['fin_1y', 'deposit_1y', 'loan_1y', 'wenzhou_rate', 'shibor_3m','reverse_repo_7d', 'mlf_6m', 'p2p_rate']

    dfr = dfr[columns]

    ds = []
    rs = []

    for i in range(0, len(dfr) - 1):

        tmp = dfr.iloc[ 0 : i + 1 ].copy()
        tmp_std = tmp.std()
        tmp_std = tmp_std[tmp_std > 0.0]

        ratio = dfr.iloc[i]
        #ratio = ratio[ratio != 0.0]

        items = tmp_std.index & ratio.index

        ds.append(dfr.index[i])

        if len(items) == 0:

            rs.append(0.0)

        else:

            tmp_std = tmp_std.loc[items]
            ratio = ratio.loc[items]

            weight = ( 1.0 / tmp_std) / (1.0 / tmp_std).sum()

            r = 0
            for item in weight.index:
                r = ratio.loc[item] * weight[item]
            rs.append(r)

    rates = pd.DataFrame(rs, index = ds, columns = ['rate'])
    print rates
    rates = ( 1 + rates ).cumprod()
    #print rates
    rates.to_csv('rates.csv')
