#coding=utf8


import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")


def beta_factor(fund_df, winda_df, rf_df)

    start = time.time()
    print 'Programme Start', start
    print '----------------------------------------------------'
    #Data Clearn

    #close      = pd.read_excel('close.xlsx', index_col='Date', parse_dates=True)        #Prices
    close      = fund_df        #Prices
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
                            returns_period[j][i:i+252] = 999

    for j in stocks:
        for i in range(len(returns_period)):
            if np.isnan(returns_period[j][i]):
                returns_period[j][i:i+252] = 999

    for j in stocks:
        returns_period[j] = np.where(returns_period[j]==999,0,1)

    #returns_period.to_excel('returns_period.xlsx')

    returns        = returns.fillna(0)
    #A              = pd.read_excel('wind_a.xlsx', index_col='Date', parse_dates=True)
    A              = winda_df
    A              = np.log(A/A.shift(1))
    A              = A[1:]
    #rf             = pd.read_excel('rf.xls', index_col='Date', parse_dates=True)
    rf             = rf_df
    rf             = (1.0 + rf/100.0)**(1.0/252.0)-1.0
    returns        = pd.merge(returns, A, left_index=True, right_index=True,how = 'left') 
    excess_returns = list(returns.columns.values)
    returns        = pd.merge(returns, rf, left_index=True, right_index=True,how = 'left') 
    returns['rf']  = returns['rf'].fillna(method='ffill')
    #returns.to_excel('returns_rf_benchmark.xlsx')
    for j in excess_returns:
        returns[j] = returns[j] - returns['rf']
    #returns.to_excel('returns_benchmark.xlsx')
    del returns['rf']

    from sklearn import linear_model
    #OLS
    def lin(x,y):
        lin  = linear_model.LinearRegression()
        reg  = lin.fit(x, y)
        coef = lin.coef_
        return coef[0]

    betas = returns.copy()
    #print betas

    for j in stocks:
        for i in range(252,len(returns.index)):
            y_tmp       = returns[j][i-252:i]
            y           = y_tmp.reshape(252,1)
            x_tmp       = returns['A'][i-252:i]
            x           = x_tmp.reshape(252,1)
            reg         = lin(x, y) 
            beta        = reg[0]
            betas[j][i] = beta
    del betas['A']
    betas = betas[252:] 
    betas.to_excel('betas.xlsx')
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    betas_value    = betas.copy()
    returns_period = returns_period[252:]
    betas          = betas * returns_period
    for j in stocks:
        betas[j] = np.where(betas[j] == 0.0, None, betas[j])

    betas['median'] = betas.median(axis=1, skipna = True)
    for j in stocks:
        betas[j] = np.where(betas[j]>betas['median'],1,0)
    del   betas['median']

    betas_high = betas.copy()
    betas_low  = 1 - betas.copy()

    betas_high     = betas_high * returns_period * betas_value
    betas_low      = betas_low * returns_period * betas_value
    returns        = returns[252:]

    for j in stocks:
        betas_high[j] = np.where(betas_high[j] == 0.0, None, betas_high[j])

    for j in stocks:
        betas_low[j]  = np.where(betas_low[j] == 0.0, None, betas_low[j])

    betas_high.to_csv('./tmp/betas_high.csv')
    betas_low.to_csv('./tmp/betas_low.csv')

    betas_high['min'] = betas_high.min(axis =1, skipna = True)
    betas_high['max'] = betas_high.max(axis =1, skipna = True)

    betas_low['min'] = betas_low.min(axis =1, skipna = True)
    betas_low['max'] = betas_low.max(axis =1, skipna = True)

    for j in stocks:
        betas_high[j] = ((betas_high[j] - betas_high['min']) / (betas_high['max'] - betas_high['min'])) + 1.0
        betas_high[j] = np.where(betas_high[j] == None, 0.0, betas_high[j])

    for j in stocks:
        betas_low[j] = ((betas_low[j] - betas_low['min']) / (betas_low['max'] - betas_low['min'])) + 1.0
        betas_low[j]  = np.where(betas_low[j] == None, 0.0, betas_low[j])

    del betas_high['min'], betas_high['max'], betas_low['min'], betas_low['max']
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    betas_high_w = betas_high.abs()
    betas_low_w  = 1.0 / betas_low.abs()

    for j in stocks:
        betas_high_w[j] = np.where(betas_high_w[j] == 0.0, None, betas_high_w[j])

    for j in stocks:
        betas_low_w[j]  = np.where(betas_low_w[j] == 0.0, None, betas_low_w[j])
        betas_low_w[j]  = np.where(betas_low_w[j] == float('inf'), None, betas_low_w[j])

    betas_high_w.to_csv('./tmp/betas_high_w.csv')
    betas_low_w.to_csv('./tmp/betas_low_w.csv')

    betas_high_w['Q95'] = betas_high_w.quantile(q=0.95, axis=1, numeric_only=False)
    betas_high_w['Q05'] = betas_high_w.quantile(q=0.05, axis=1, numeric_only=False)

    for j in stocks:
        betas_high_w[j] = np.where(betas_high_w[j] > betas_high_w['Q95'], betas_high_w['Q95'], betas_high_w[j])
        betas_high_w[j] = np.where(betas_high_w[j] < betas_high_w['Q05'], betas_high_w['Q05'], betas_high_w[j])
    del betas_high_w['Q95'], betas_high_w['Q05']

    betas_high_w['sum'] = betas_high_w.sum(axis =1, skipna = True)
    for j in stocks:
        betas_high_w[j] = betas_high_w[j]/betas_high_w['sum']
    del betas_high_w['sum']

    betas_high_w = betas_high_w.shift(1)

    high_beta_returns        = betas_high_w * returns
    high_beta_returns['sum'] = high_beta_returns.sum(axis =1, skipna = True)
    high_beta_returns['sum'] = high_beta_returns['sum'].fillna(0)
    high_beta_returns['uv']  = 1.0
    for i in range(1,len(high_beta_returns.index)):
        high_beta_returns['uv'][i] =  high_beta_returns['uv'][i-1]*(1.0+high_beta_returns['sum'][i])

    high_beta_returns.to_csv('./tmp/high_beta_returns.csv')

    betas_low_w['Q95'] = betas_low_w.quantile(q=0.95, axis=1, numeric_only=False)
    betas_low_w['Q05'] = betas_low_w.quantile(q=0.05, axis=1, numeric_only=False)

    for j in stocks:
        betas_low_w[j] = np.where(betas_low_w[j] > betas_low_w['Q95'], betas_low_w['Q95'], betas_low_w[j])
        betas_low_w[j] = np.where(betas_low_w[j] < betas_low_w['Q05'], betas_low_w['Q05'], betas_low_w[j])
    del betas_low_w['Q95'], betas_low_w['Q05']

    betas_low_w['sum'] = betas_low_w.sum(axis =1, skipna = True)
    for j in stocks:
        betas_low_w[j] = betas_low_w[j]/betas_low_w['sum']
    del betas_low_w['sum']

    betas_low_w = betas_low_w.shift(1)

    low_beta_returns        = betas_low_w * returns
    low_beta_returns['sum'] = low_beta_returns.sum(axis =1, skipna = True)
    low_beta_returns['sum'] = low_beta_returns['sum'].fillna(0)
    low_beta_returns['uv']  = 1.0
    for i in range(1,len(low_beta_returns.index)):
        low_beta_returns['uv'][i] =  low_beta_returns['uv'][i-1]*(1.0+low_beta_returns['sum'][i])

    low_beta_returns.to_csv('low_beta_returns.csv')


    end = time.time()
    print 'Programme End', end
    print 'Costs (Mins)', (end-start)/60.0	 
    print '----------------------------------------------------'

    return high_beta_returns, low_beta_returns
