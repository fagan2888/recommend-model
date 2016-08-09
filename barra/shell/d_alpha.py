#coding=utf8


import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")


def alpha(fund_df, winda_df, rf_df):

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
        alpha = reg.intercept_
        return alpha

    alphas = returns.copy()
    #print alphas

    for j in stocks:
        for i in range(252,len(returns.index)):
            y_tmp       = returns[j][i-252:i]
            y           = y_tmp.reshape(252,1)
            x_tmp       = returns['A'][i-252:i]
            x           = x_tmp.reshape(252,1)
            reg         = lin(x, y) 
            alpha        = reg[0]
            alphas[j][i] = alpha
    del alphas['A']
    alphas = alphas[252:] 
    alphas.to_excel('alphas.xlsx')
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    alphas_value    = alphas.copy()
    returns_period = returns_period[252:]
    alphas          = alphas * returns_period
    for j in stocks:
        alphas[j] = np.where(alphas[j] == 0.0, None, alphas[j])

    alphas['median'] = alphas.median(axis=1, skipna = True)
    for j in stocks:
        alphas[j] = np.where(alphas[j]>alphas['median'],1,0)
    del   alphas['median']

    alphas_high = alphas.copy()
    alphas_low  = 1 - alphas.copy()

    alphas_high     = alphas_high * returns_period * alphas_value
    alphas_low      = alphas_low * returns_period * alphas_value
    returns        = returns[252:]

    for j in stocks:
        alphas_high[j] = np.where(alphas_high[j] == 0.0, None, alphas_high[j])

    for j in stocks:
        alphas_low[j]  = np.where(alphas_low[j] == 0.0, None, alphas_low[j])

    #alphas_high.to_excel('alphas_high.xlsx')
    #alphas_low.to_excel('alphas_low.xlsx')

    alphas_high['min'] = alphas_high.min(axis =1, skipna = True)
    alphas_high['max'] = alphas_high.max(axis =1, skipna = True)

    alphas_low['min'] = alphas_low.min(axis =1, skipna = True)
    alphas_low['max'] = alphas_low.max(axis =1, skipna = True)

    for j in stocks:
        alphas_high[j] = ((alphas_high[j] - alphas_high['min']) / (alphas_high['max'] - alphas_high['min'])) + 1.0
        alphas_high[j] = np.where(alphas_high[j] == None, 0.0, alphas_high[j])

    for j in stocks:
        alphas_low[j] = ((alphas_low[j] - alphas_low['min']) / (alphas_low['max'] - alphas_low['min'])) + 1.0
        alphas_low[j]  = np.where(alphas_low[j] == None, 0.0, alphas_low[j])

    del alphas_high['min'], alphas_high['max'], alphas_low['min'], alphas_low['max']
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    alphas_high_w = alphas_high.abs()
    alphas_low_w  = 1.0 / alphas_low.abs()

    for j in stocks:
        alphas_high_w[j] = np.where(alphas_high_w[j] == 0.0, None, alphas_high_w[j])

    for j in stocks:
        alphas_low_w[j]  = np.where(alphas_low_w[j] == 0.0, None, alphas_low_w[j])
        alphas_low_w[j]  = np.where(alphas_low_w[j] == float('inf'), None, alphas_low_w[j])

    #alphas_high_w.to_excel('alphas_high_w.xlsx')
    #alphas_low_w.to_excel('alphas_low_w.xlsx')

    alphas_high_w['Q95'] = alphas_high_w.quantile(q=0.95, axis=1, numeric_only=False)
    alphas_high_w['Q05'] = alphas_high_w.quantile(q=0.05, axis=1, numeric_only=False)

    for j in stocks:
        alphas_high_w[j] = np.where(alphas_high_w[j] > alphas_high_w['Q95'], alphas_high_w['Q95'], alphas_high_w[j])
        alphas_high_w[j] = np.where(alphas_high_w[j] < alphas_high_w['Q05'], alphas_high_w['Q05'], alphas_high_w[j])
    del alphas_high_w['Q95'], alphas_high_w['Q05']

    alphas_high_w['sum'] = alphas_high_w.sum(axis =1, skipna = True)
    for j in stocks:
        alphas_high_w[j] = alphas_high_w[j]/alphas_high_w['sum']
    del alphas_high_w['sum']

    alphas_high_w = alphas_high_w.shift(1)

    high_alpha_returns        = alphas_high_w * returns
    high_alpha_returns['sum'] = high_alpha_returns.sum(axis =1, skipna = True)
    high_alpha_returns['sum'] = high_alpha_returns['sum'].fillna(0)
    high_alpha_returns['uv']  = 1.0
    for i in range(1,len(high_alpha_returns.index)):
        high_alpha_returns['uv'][i] =  high_alpha_returns['uv'][i-1]*(1.0+high_alpha_returns['sum'][i])

    high_alpha_returns.to_csv('./tmp/high_alpha_returns.csv')

    alphas_low_w['Q95'] = alphas_low_w.quantile(q=0.95, axis=1, numeric_only=False)
    alphas_low_w['Q05'] = alphas_low_w.quantile(q=0.05, axis=1, numeric_only=False)

    for j in stocks:
        alphas_low_w[j] = np.where(alphas_low_w[j] > alphas_low_w['Q95'], alphas_low_w['Q95'], alphas_low_w[j])
        alphas_low_w[j] = np.where(alphas_low_w[j] < alphas_low_w['Q05'], alphas_low_w['Q05'], alphas_low_w[j])
    del alphas_low_w['Q95'], alphas_low_w['Q05']

    alphas_low_w['sum'] = alphas_low_w.sum(axis =1, skipna = True)
    for j in stocks:
        alphas_low_w[j] = alphas_low_w[j]/alphas_low_w['sum']
    del alphas_low_w['sum']

    alphas_low_w = alphas_low_w.shift(1)

    low_alpha_returns        = alphas_low_w * returns
    low_alpha_returns['sum'] = low_alpha_returns.sum(axis =1, skipna = True)
    low_alpha_returns['sum'] = low_alpha_returns['sum'].fillna(0)
    low_alpha_returns['uv']  = 1.0
    for i in range(1,len(low_alpha_returns.index)):
        low_alpha_returns['uv'][i] =  low_alpha_returns['uv'][i-1]*(1.0+low_alpha_returns['sum'][i])

    low_alpha_returns.to_csv('./tmp/low_alpha_returns.csv')


    end = time.time()
    print 'Programme End', end
    print 'Costs (Mins)', (end-start)/60.0	 
    print '----------------------------------------------------'

    return high_alpha_returns, low_alpha_returns
