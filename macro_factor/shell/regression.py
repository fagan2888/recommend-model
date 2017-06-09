#coding=utf8


import pandas as pd
import numpy as np
import statsmodels.api as sm
import indicator_delay
from sklearn import linear_model
import sys


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
    df = df.fillna(method = 'pad')
    col = ['deposit_rate','gdp_yoy', '000300.pe', '000852.pe', '3mtbr', '1ytbr', '10ytbr', '2ytbr', 'shibor', 'm2', 'm2_yoy', '000852.SH', '000300.SH']


    tmp_df = df[col]
    print tmp_df
    tmp_df.to_csv('tmp.csv')


    '''
    del df['town_unemployment_insurance_num']
    del df['core_cpi_yoy']
    del df['investor_confidence_index']
    del df['rpi_yoy']
    del df['agricultural_production_price_index_yoy']
    del df['m1_yoy_m2_yoy']
    cols = df.columns[0: -9]
    #cols = df.columns[2: 3]
    #cols = df.columns[4: 5]
    #cols = ['rpi_yoy']
    #cols = df.columns[38]
    index_col = '000852.SH'
    zz1000 = df[index_col]
    zz1000 = zz1000.dropna()
    zz1000 = zz1000.rolling(window = 6).apply(lambda x :  x[-1] / x[0] - 1)
    zz1000 = zz1000.shift(-6)
    zz1000 = zz1000.dropna()
    #zz1000 = df[index_col].shift().dropna()
    #zz1000 = df[index_col].fillna(method = 'pad').dropna()
    df = df[cols]
    df = df.loc[zz1000.index]
    df = df.fillna(method = 'pad').dropna()
    zz1000 = zz1000.loc[df.index]


    dates = zz1000.index

    total_num = 0
    mov_correct = 0
    X = df.pct_change().fillna(0.0).values
    X = sm.add_constant(X)
    y = zz1000.values


    #X = df.values
    #X = sm.add_constant(X)
    #y = zz1000.values
    errors = 0
    for i in range(100, len(df)):
        train_X = X[i - 100 : i]
        train_y = y[i - 100 : i]
        test_X = X[i].reshape(1, -1)
        test_y = y[i]

        model = linear_model.LinearRegression()
        model.fit(train_X, train_y)

        predict = model.predict(test_X)
        print dates[i], predict, test_y
        total_num = total_num + 1.0
        if test_y >= 0 and predict >= 0:
            mov_correct += 1
        elif test_y <= 0 and predict <= 0:
            mov_correct += 1

        errors += (predict - test_y) ** 2

        #model = sm.OLS(y, X)
        #result = model.fit()

        #print result.summary()
    print mov_correct / total_num, (errors / total_num) ** 0.5


    #print zz1000.index

    #print df.index
    #X = df.values
    #X = sm.add_constant(X)
    #y = zz1000.values

    #print len(df)

    #X = df.values
    #X = sm.add_constant(X)
    #y = zz1000.values
    #model = sm.OLS(y, X)
    #result = model.fit()
    #print result.summary()


    #X = df.pct_change().dropna().values
    #X = sm.add_constant(X)
    #y = zz1000.pct_change().dropna().values
    #model = sm.OLS(y, X)
    #result = model.fit()
    #print result.summary()
    '''
