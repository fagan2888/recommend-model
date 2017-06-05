#coding=utf8


import pandas as pd
import numpy as np
import statsmodels.api as sm
import indicator_delay


#def factor_delay_index_ahead_process(factor, index_price, factor_delay):
#
#    factor = factor.shift(factor_delay)
#    factor = factor.dropna()
#    index_price = index_price.shift(-1)
#    index_price = index_price.dropna()
#    index = index_price.index & factor.index
#    factor = factor.loc[index]
#    index_price = index_price.loc[index]
#
#    return factor, index_price


if __name__ == '__main__':

    df = pd.read_csv('./data/macro_factor_index.csv', index_col = ['date'], parse_dates = ['date'])
    delay = indicator_delay.delay
    cols = df.columns[0: -9]
    #cols = df.columns[2: 3]
    #cols = df.columns[4: 5]
    #cols = ['rpi_yoy']
    index_col = '000852.SH'
    zz1000 = df[index_col].shift(-1).dropna()
    #zz1000 = df[index_col].fillna(method = 'pad').dropna()
    df = df[cols]
    df = df.loc[zz1000.index]
    df = df.fillna(method = 'pad').dropna()
    zz1000 = zz1000.loc[df.index]
    #print df

    X = df.values
    X = sm.add_constant(X)
    y = zz1000.values
    model = sm.OLS(y, X)
    result = model.fit()
    print result.summary()

    #X = df.pct_change().dropna().values
    #X = sm.add_constant(X)
    #y = zz1000.pct_change().dropna().values
    #model = sm.OLS(y, X)
    #result = model.fit()
    #print result.summary()



