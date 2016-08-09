#coding=utf8


import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")


def size_factor(fund_df, size_df):


    start = time.time()
    print 'Programme Start', start
    print '----------------------------------------------------'
    #Data Clearn

    #close      = pd.read_excel('../../data/close.xlsx', index_col='Date', parse_dates=True)        #Prices
    close      = fund_df        #Prices
    returns    = np.log(close / close.shift(1)).fillna(0)
    stock_list = set(list(returns.columns.values))
    #size       = pd.read_csv('../../data/size.csv', index_col='Date', parse_dates=True)            #Size
    size       = size_df       #Size
    size_list  = set(list(size.columns.values))
    col        = size_list & stock_list
    size       = size[list(col)]
    returns    = returns[list(col)]
    #size.to_excel('size.xlsx')
    #returns.to_excel('returns_size.xlsx')

    stocks     = list(returns.columns.values)
    for j in stocks:
        returns[j] = np.where(returns[j] == float('inf'), 0.0, returns[j])
        returns[j] = np.where(returns[j] == float('-inf'), 0.0, returns[j])

    #returns.to_excel('returns_size.xlsx')

    size_mark             = size.copy()
    size_mark['median']   = size_mark.median(axis=1, skipna=True)
    for j in stocks:
        size_mark[j]      = np.where(size_mark[j]<size_mark['median'],0,1)
    del size_mark['median']
    big_size   = size_mark.copy()
    small_size = 1 - size_mark.copy()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    big_size   = big_size * size
    small_size = small_size * size

    #big_size.to_excel('big_size.xlsx')
    #small_size.to_excel('small_size.xlsx')

    big_size_weight   = big_size.copy()
    small_size_weight = small_size.copy()

    big_size_weight    = big_size_weight.abs()
    small_size_weight  = 1.0 / small_size_weight.abs()

    for j in stocks:
        big_size_weight[j] = np.where(big_size_weight[j] == 0.0, None, big_size_weight[j])

    for j in stocks:
        small_size_weight[j]  = np.where(small_size_weight[j] == 0.0, None, small_size_weight[j])
        small_size_weight[j]  = np.where(small_size_weight[j] == float('inf'), None, small_size_weight[j])

    big_size_weight['Q95'] = big_size_weight.quantile(q=0.95, axis=1, numeric_only=False)
    big_size_weight['Q05'] = big_size_weight.quantile(q=0.05, axis=1, numeric_only=False)


    for j in stocks:
        big_size_weight[j] = np.where(big_size_weight[j] > big_size_weight['Q95'], big_size_weight['Q95'], big_size_weight[j])
        big_size_weight[j] = np.where(big_size_weight[j] < big_size_weight['Q05'], big_size_weight['Q05'], big_size_weight[j])
    del big_size_weight['Q95'], big_size_weight['Q05']

    big_size_weight['sum'] = big_size_weight.sum(axis =1, skipna = True)

    for j in stocks:
        big_size_weight[j] = big_size_weight[j]/big_size_weight['sum']
    del big_size_weight['sum']


    big_size_weight = big_size_weight.shift(1)

    big_size_returns        = big_size_weight * returns
    big_size_returns['sum'] = big_size_returns.sum(axis =1, skipna = True)
    big_size_returns['sum'] = big_size_returns['sum'].fillna(0)
    big_size_returns['uv']  = 1.0
    for i in range(1,len(big_size_returns.index)):
        big_size_returns['uv'][i] =  big_size_returns['uv'][i-1]*(1.0+big_size_returns['sum'][i])

    big_size_returns.to_csv('./tmp/big_size_returns.csv')

    small_size_weight['Q95'] = small_size_weight.quantile(q=0.95, axis=1, numeric_only=False)
    small_size_weight['Q05'] = small_size_weight.quantile(q=0.05, axis=1, numeric_only=False)


    for j in stocks:
        small_size_weight[j] = np.where(small_size_weight[j] > small_size_weight['Q95'], small_size_weight['Q95'], small_size_weight[j])
        small_size_weight[j] = np.where(small_size_weight[j] < small_size_weight['Q05'], small_size_weight['Q05'], small_size_weight[j])
    del small_size_weight['Q95'], small_size_weight['Q05']

    small_size_weight['sum'] = small_size_weight.sum(axis =1, skipna = True)
    for j in stocks:
        small_size_weight[j] = small_size_weight[j]/small_size_weight['sum']
    del small_size_weight['sum']


    small_size_weight = small_size_weight.shift(1)

    small_size_returns        = small_size_weight * returns
    small_size_returns['sum'] = small_size_returns.sum(axis =1, skipna = True)
    small_size_returns['sum'] = small_size_returns['sum'].fillna(0)

    small_size_returns['uv']  = 1.0
    for i in range(1,len(small_size_returns.index)):
        small_size_returns['uv'][i] =  small_size_returns['uv'][i-1]*(1.0+small_size_returns['sum'][i])

    small_size_returns.to_csv('./tmp/small_size_returns.csv')

    end = time.time()
    print 'Programme End', end
    print 'Costs (Mins)', (end-start)/60.0
    print '----------------------------------------------------'

    return big_size_returns, small_size_returns
