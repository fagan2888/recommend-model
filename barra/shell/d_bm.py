#coding=utf8


import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")


def bm(fund_df, pb_df):

    start = time.time()
    print 'Programme Start', start
    print '----------------------------------------------------'
    #Data Clearn

    #close      = pd.read_excel('close.xlsx', index_col='Date', parse_dates=True)        #Prices
    close      = fund_df
    returns    = np.log(close / close.shift(1)).fillna(0)
    stock_list = set(list(returns.columns.values))
    #pb         = pd.read_csv('pb.csv', index_col='Date', parse_dates=True)                #pb
    pb         = pb_df
    bm         = 1.0 / pb
    #print bm
    bm_list    = set(list(bm.columns.values))
    col        = bm_list & stock_list
    bm         = bm[list(col)]
    returns    = returns[list(col)]
    #bm.to_excel('bm.xlsx')
    #returns.to_excel('returns_bm.xlsx')

    stocks     = list(returns.columns.values)
    for j in stocks:
        returns[j] = np.where(returns[j] == float('inf'), 0.0, returns[j])
        returns[j] = np.where(returns[j] == float('-inf'), 0.0, returns[j])

    #returns.to_excel('returns_bm.xlsx')

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    bm_mark             = bm.copy() 
    bm_mark['median']   = bm_mark.median(axis=1, skipna=True)
    for j in stocks:
        bm_mark[j]      = np.where(bm_mark[j]<bm_mark['median'],0,1)
    del bm_mark['median']
    big_bm   = bm_mark.copy()
    small_bm = 1 - bm_mark.copy()

    for j in stocks:
        big_bm[j] = np.where(big_bm[j] == 0.0, None, big_bm[j])

    for j in stocks:
        small_bm[j] = np.where(small_bm[j] == 0.0, None, small_bm[j])

    big_bm   = big_bm * bm
    small_bm = small_bm * bm

    big_bm['min'] = big_bm.min(axis =1, skipna = True)
    big_bm['max'] = big_bm.max(axis =1, skipna = True)

    small_bm['min'] = small_bm.min(axis =1, skipna = True)
    small_bm['max'] = small_bm.max(axis =1, skipna = True)

    for j in stocks:
        big_bm[j] = ((big_bm[j] - big_bm['min']) / (big_bm['max'] - big_bm['min'])) + 1.0
        big_bm[j] = np.where(big_bm[j] == None, 0.0, big_bm[j])

    for j in stocks:
        small_bm[j] = ((small_bm[j] - small_bm['min']) / (small_bm['max'] - small_bm['min'])) + 1.0
        small_bm[j] = np.where(small_bm[j] == None, 0.0, small_bm[j])

    del big_bm['min'], big_bm['max'], small_bm['min'], small_bm['max']
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #big_bm.to_excel('big_bm.xlsx')
    #small_bm.to_excel('small_bm.xlsx')

    big_bm_weight   = big_bm.copy()
    small_bm_weight = small_bm.copy()

    big_bm_weight    = big_bm_weight.abs()
    small_bm_weight  = 1.0 / small_bm_weight.abs()

    for j in stocks:
        big_bm_weight[j] = np.where(big_bm_weight[j] == 0.0, None, big_bm_weight[j])

    for j in stocks:
        small_bm_weight[j]  = np.where(small_bm_weight[j] == 0.0, None, small_bm_weight[j])
        small_bm_weight[j]  = np.where(small_bm_weight[j] == float('inf'), None, small_bm_weight[j])

    big_bm_weight['Q95'] = big_bm_weight.quantile(q=0.95, axis=1, numeric_only=False)
    big_bm_weight['Q05'] = big_bm_weight.quantile(q=0.05, axis=1, numeric_only=False)

    for j in stocks:
        big_bm_weight[j] = np.where(big_bm_weight[j] > big_bm_weight['Q95'], big_bm_weight['Q95'], big_bm_weight[j])
        big_bm_weight[j] = np.where(big_bm_weight[j] < big_bm_weight['Q05'], big_bm_weight['Q05'], big_bm_weight[j])
    del big_bm_weight['Q95'], big_bm_weight['Q05']

    big_bm_weight['sum'] = big_bm_weight.sum(axis =1, skipna = True)
    for j in stocks:
        big_bm_weight[j] = big_bm_weight[j]/big_bm_weight['sum']
    del big_bm_weight['sum']

    big_bm_weight = big_bm_weight.shift(1)

    big_bm_returns        = big_bm_weight * returns
    big_bm_returns['sum'] = big_bm_returns.sum(axis =1, skipna = True)
    big_bm_returns['sum'] = big_bm_returns['sum'].fillna(0)

    big_bm_returns['uv']  = 1.0
    for i in range(1,len(big_bm_returns.index)):
        big_bm_returns['uv'][i] =  big_bm_returns['uv'][i-1]*(1.0+big_bm_returns['sum'][i])

    big_bm_returns.to_csv('./tmp/big_bm_returns.csv')

    small_bm_weight['Q95'] = small_bm_weight.quantile(q=0.95, axis=1, numeric_only=False)
    small_bm_weight['Q05'] = small_bm_weight.quantile(q=0.05, axis=1, numeric_only=False)

    for j in stocks:
        small_bm_weight[j] = np.where(small_bm_weight[j] > small_bm_weight['Q95'], small_bm_weight['Q95'], small_bm_weight[j])
        small_bm_weight[j] = np.where(small_bm_weight[j] < small_bm_weight['Q05'], small_bm_weight['Q05'], small_bm_weight[j])
    del small_bm_weight['Q95'], small_bm_weight['Q05']

    small_bm_weight['sum'] = small_bm_weight.sum(axis =1, skipna = True)
    for j in stocks:
        small_bm_weight[j] = small_bm_weight[j]/small_bm_weight['sum']
    del small_bm_weight['sum']

    small_bm_weight = small_bm_weight.shift(1)

    small_bm_returns        = small_bm_weight * returns
    small_bm_returns['sum'] = small_bm_returns.sum(axis =1, skipna = True)
    small_bm_returns['sum'] = small_bm_returns['sum'].fillna(0)

    small_bm_returns['uv']  = 1.0
    for i in range(1,len(small_bm_returns.index)):
        small_bm_returns['uv'][i] =  small_bm_returns['uv'][i-1]*(1.0+small_bm_returns['sum'][i])

    small_bm_returns.to_csv('./tmp/small_bm_returns.csv')

    end = time.time()
    print 'Programme End', end
    print 'Costs (Mins)', (end-start)/60.0	 
    print '----------------------------------------------------'

    return big_bm_returns, small_bm_returns
