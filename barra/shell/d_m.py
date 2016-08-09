#coding=utf8

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")


def momentum(fund_df):

    start = time.time()
    print 'Programme Start', start
    print '----------------------------------------------------'
    #Data Clearn

    #close      = pd.read_excel('close.xlsx', index_col='Date', parse_dates=True)        #Prices
    close      = fund_df 
    returns    = np.log(close / close.shift(1))
    returns    = returns[1:]

    stocks     = list(returns.columns.values)
    for j in stocks:
        returns[j] = np.where(returns[j] == float('inf'), 0.0, returns[j])
        returns[j] = np.where(returns[j] == float('-inf'), 0.0, returns[j])

    #returns.to_excel('returns.xlsx')

    returns_period = returns.copy()

    for j in stocks:
        count = 0
        for i in range(len(returns_period.index)-1):
            if returns_period[j][i] == 0:
                if returns_period[j][i+1] == 0:
                    if returns_period[j][i] == returns_period[j][i+1]:
                        count += 1
                        if count >= 7:
                            returns_period[j][i:i+504] = 999

    for j in stocks:
        for i in range(len(returns_period)):
            if np.isnan(returns_period[j][i]):
                returns_period[j][i:i+504] = 999

    for j in stocks:
        returns_period[j] = np.where(returns_period[j]==999,0,1)

    #returns_period.to_excel('returns_period.xlsx')

    rf             = pd.read_excel('rf.xls', index_col='Date', parse_dates=True)
    rf             = (1.0 + rf/100.0)**(1.0/252.0)-1.0
    returns        = pd.merge(returns, rf, left_index=True, right_index=True,how = 'left') 
    for j in stocks:
        returns[j] = returns[j] - returns['rf']
    del returns['rf']

    momentum       = pd.rolling_sum(returns,21)
    momentum       = momentum[504:]
    returns        = returns[504:]
    returns_period = returns_period[504:]
    #momentum.to_excel('momentum_21.xlsx')
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    momentum_mark = momentum.copy()
    momentum_mark = momentum_mark * returns_period
    for j in stocks:
        momentum_mark[j] = np.where(momentum_mark[j] == 0.0, None, momentum_mark[j])

    momentum_mark['median'] = momentum_mark.median(axis =1, skipna = True)
    for j in stocks:
        momentum_mark[j] = np.where(momentum_mark[j] > momentum_mark['median'], 1,0)
    del momentum_mark['median']

    high_m = momentum_mark.copy()
    low_m  = 1.0 - high_m.copy()

    high_m = high_m * momentum * returns_period
    low_m  = low_m * momentum * returns_period


    for j in stocks:
        high_m[j] = np.where(high_m[j] == 0.0, None, high_m[j])

    for j in stocks:
        low_m[j]  = np.where(low_m[j] == 0.0, None, low_m[j])

    high_m['min'] = high_m.min(axis =1, skipna = True)
    high_m['max'] = high_m.max(axis =1, skipna = True)

    low_m['min'] = low_m.min(axis =1, skipna = True)
    low_m['max'] = low_m.max(axis =1, skipna = True)


    for j in stocks:
        high_m[j] = ((high_m[j]-high_m['min'])/(low_m['max']-high_m['min'])) + 1.0
        high_m[j] = np.where(high_m[j] == None, 0.0, high_m[j])

    for j in stocks:
        low_m[j] = ((low_m[j]-low_m['min'])/(low_m['max']-low_m['min'])) + 1.0
        low_m[j]  = np.where(low_m[j] == None, 0.0, low_m[j])

    del high_m['min'], high_m['max'], low_m['min'], low_m['max']
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #high_m.to_excel('high_m.xlsx')
    #low_m.to_excel('low_m.xlsx')

    high_m_w = high_m.abs()
    low_m_w  = 1.0 / low_m.abs()

    for j in stocks:
        high_m_w[j] = np.where(high_m_w[j] == 0.0, None, high_m_w[j])

    for j in stocks:
        low_m_w[j]  = np.where(low_m_w[j] == 0.0, None, low_m_w[j])
        low_m_w[j]  = np.where(low_m_w[j] == float('inf'), None, low_m_w[j])

    #high_m_w.to_excel('high_m_w.xlsx')
    #low_m_w.to_excel('low_m_w.xlsx')

    high_m_w['Q95'] = high_m_w.quantile(q=0.95, axis=1, numeric_only=False)
    high_m_w['Q05'] = high_m_w.quantile(q=0.05, axis=1, numeric_only=False)

    for j in stocks:
        high_m_w[j] = np.where(high_m_w[j] > high_m_w['Q95'], high_m_w['Q95'], high_m_w[j])
        high_m_w[j] = np.where(high_m_w[j] < high_m_w['Q05'], high_m_w['Q05'], high_m_w[j])
    del high_m_w['Q95'], high_m_w['Q05']

    high_m_w['sum'] = high_m_w.sum(axis =1, skipna = True)
    for j in stocks:
        high_m_w[j] = high_m_w[j]/high_m_w['sum']
    del high_m_w['sum']

    high_m_w = high_m_w.shift(1)

    high_momentum_returns        = high_m_w * returns
    high_momentum_returns['sum'] = high_momentum_returns.sum(axis =1, skipna = True)
    high_momentum_returns['sum'] = high_momentum_returns['sum'].fillna(0)

    high_momentum_returns['uv']  = 1.0
    for i in range(1,len(high_momentum_returns.index)):
        high_momentum_returns['uv'][i] =  high_momentum_returns['uv'][i-1]*(1.0+high_momentum_returns['sum'][i])

    high_momentum_returns.to_csv('./tmp/high_momentum_returns.csv')


    low_m_w['Q95'] = low_m_w.quantile(q=0.95, axis=1, numeric_only=False)
    low_m_w['Q05'] = low_m_w.quantile(q=0.05, axis=1, numeric_only=False)

    for j in stocks:
        low_m_w[j] = np.where(low_m_w[j] > low_m_w['Q95'], low_m_w['Q95'], low_m_w[j])
        low_m_w[j] = np.where(low_m_w[j] < low_m_w['Q05'], low_m_w['Q05'], low_m_w[j])
    del low_m_w['Q95'], low_m_w['Q05']

    low_m_w['sum'] = low_m_w.sum(axis =1, skipna = True)
    for j in stocks:
        low_m_w[j] = low_m_w[j]/low_m_w['sum']
    del low_m_w['sum']

    low_m_w = low_m_w.shift(1)

    low_momentum_returns        = low_m_w * returns
    low_momentum_returns['sum'] = low_momentum_returns.sum(axis =1, skipna = True)
    low_momentum_returns['sum'] = low_momentum_returns['sum'].fillna(0)

    low_momentum_returns['uv']  = 1.0
    for i in range(1,len(low_momentum_returns.index)):
        low_momentum_returns['uv'][i] =  low_momentum_returns['uv'][i-1]*(1.0+low_momentum_returns['sum'][i])


    low_momentum_returns.to_csv('./tmp/low_momentum_returns.csv')

    end = time.time()
    print 'Programme End', end
    print 'Costs (Mins)', (end-start)/60.0	 
    print '----------------------------------------------------'

    return high_momentum_returns, low_momentum_returns
