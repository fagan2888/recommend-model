#coding=utf8



import pandas as pd
import string
import statsmodels.api as sm
import numpy as np
import time
import sys
sys.path.append('shell/factor')
import ht_factor



if __name__ == '__main__':


    stock_df = pd.read_csv('./data/stock_price_adjust.csv', index_col = 'date', parse_dates = ['date'])
    index_df = pd.read_csv('./data/index_price.csv', index_col = 'date', parse_dates = ['date'])
    index_df = index_df[['000300']]
    index_dfr = index_df.pct_change().fillna(0.0)


    beta_df = pd.read_csv('./data/std_beta.csv', parse_dates = ['date'], index_col = 'date')
    market_value_df = pd.read_csv('./data/std_market_value.csv', parse_dates = ['date'], index_col = 'date')
    momentum_df = pd.read_csv('./data/std_momentum.csv', parse_dates = ['date'], index_col = 'date')
    dastd_df = pd.read_csv('./data/std_dastd.csv', parse_dates = ['date'], index_col = 'date')
    bp_df = pd.read_csv('./data/std_bp.csv', parse_dates = ['date'], index_col = 'date')
    liquidity_df = pd.read_csv('./data/std_liquidity.csv', parse_dates = ['date'], index_col = 'date')


    '''
    beta_df = beta_df.iloc[-1010:]
    market_value_df = market_value_df.iloc[-1010:]
    bp_df = bp_df.iloc[-1010:]
    momentum_df = momentum_df.iloc[-1010:]
    dastd_df = dastd_df.iloc[-1010:]
    liquidity_df = liquidity_df.iloc[-1010:]

    beta_df = ht_factor.factor_standard(beta_df)
    print 'beta done'
    market_value_df = ht_factor.factor_standard(market_value_df)
    print 'market value done'
    momentum_df = ht_factor.factor_standard(momentum_df)
    print 'momentum done'
    dastd_df = ht_factor.factor_standard(dastd_df)
    print 'dastd done'
    bp_df = ht_factor.factor_standard(bp_df)
    print 'bp done'
    liquidity_df = ht_factor.factor_standard(liquidity_df)
    print 'liquidity done'

    beta_df.to_csv('./data/std_beta.csv')
    market_value_df.to_csv('./data/std_market_value.csv')
    momentum_df.to_csv('./data/std_momentum.csv')
    dastd_df.to_csv('./data/std_dastd.csv')
    bp_df.to_csv('./data/std_bp.csv')
    liquidity_df.to_csv('./data/std_liquidity.csv')
    '''

    stock_dfr = stock_df.pct_change().fillna(0.0)
    beta_dfr = beta_df.pct_change().fillna(0.0)
    market_value_dfr = market_value_df.pct_change().fillna(0.0)
    bp_dfr = bp_df.pct_change().fillna(0.0)
    momentum_dfr = momentum_df.pct_change().fillna(0.0)
    dastd_dfr = dastd_df.pct_change().fillna(0.0)
    liquidity_dfr = liquidity_df.pct_change().fillna(0.0)

    stock_dfr = stock_dfr.iloc[-1000:]
    beta_dfr = beta_dfr.iloc[-1000:]
    market_value_dfr = market_value_dfr.iloc[-1000:]
    bp_dfr = bp_dfr.iloc[-1000:]
    momentum_dfr = momentum_dfr.iloc[-1000:]
    dastd_dfr = dastd_dfr.iloc[-1000:]
    liquidity_dfr = liquidity_dfr.iloc[-1000:]


    codes = stock_dfr.columns & market_value_dfr.columns & bp_dfr.columns & momentum_dfr.columns & dastd_dfr.columns & liquidity_dfr.columns
    dates = index_dfr.index & stock_dfr.index & market_value_dfr.index & bp_dfr.index & momentum_dfr.index & dastd_dfr.index & liquidity_dfr.index


    stock_dfr = stock_dfr[codes]
    market_value_dfr = market_value_dfr[codes]
    bp_dfr = bp_dfr[codes]
    momentum_dfr = momentum_dfr[codes]
    dastd_dfr = dastd_dfr[codes]
    liquidity_dfr = liquidity_dfr[codes]


    stock_dfr = stock_dfr.loc[dates]
    market_value_dfr = market_value_dfr.loc[dates]
    bp_dfr = bp_dfr.loc[dates]
    momentum_dfr = momentum_dfr.loc[dates]
    dastd_dfr = dastd_dfr.loc[dates]
    liquidity_dfr = liquidity_dfr.loc[dates]

    #print stock_dfr.columns
    #print momentum_dfr.columns

    for d in dates:
        index_r = index_dfr.loc[d].values[0]
        y = []
        X = []
        for code in stock_dfr.columns:
            stock_r = stock_dfr.loc[d, code]
            market_value_r = market_value_dfr.loc[d, code]
            y.append(stock_r)
            X.append([index_r, market_value_r])
            #print d, code, market_value_r
        X = sm.add_constant(X)
        model  = sm.OLS(y,X)
        result = model.fit()
        rsquared_adj = result.rsquared_adj
        params = result.params
        print d, rsquared_adj, result.params

    #print momentum_dfr
    #print
