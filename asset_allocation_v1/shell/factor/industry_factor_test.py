#coding=utf8


import pandas as pd
import numpy as np
import statsmodels.api as sm


if __name__ == '__main__':


    stock_industrycode_df = pd.read_csv('./data/stock_industryindex.csv', dtype = {'SYMBOL':str, 'INDEXSYMBOL':str})
    stock_industrycode_df = stock_industrycode_df.set_index('SYMBOL')
    index_df = pd.read_csv('./data/index_price.csv', index_col = ['date'], parse_dates = ['date'])
    #stock_market_value_df = pd.read_csv('./data/stock_market_value.csv', index_col = 'date', parse_dates = ['date'])
    stock_df              = pd.read_csv('./data/stock_price_adjust.csv', index_col = 'date', parse_dates = ['date'])
    industry_index_df     = pd.read_csv('./data/industry_index.csv', index_col = ['date'], parse_dates = ['date'])
    #beta_df = pd.read_csv('./data/beta_index.csv', parse_dates = ['date'], index_col = 'date')
    market_value_df = pd.read_csv('./data/market_value.csv', parse_dates = ['date'], index_col = 'date')
    beta_df = pd.read_csv('./data/beta.csv', parse_dates = ['date'], index_col = 'date')
    codes = stock_industrycode_df.index & stock_df.columns & market_value_df.columns & beta_df.columns
    #print market_value_df.columns
    #print codes
    stock_industrycode_df = stock_industrycode_df.loc[codes]
    stock_df = stock_df[codes]
    market_value_df = market_value_df[codes]
    #momentum_df = pd.read_csv('./data/momentum_index.csv', parse_dates = ['date'], index_col = 'date')
    #dastd_df = pd.read_csv('./data/dastd_index.csv', parse_dates = ['date'], index_col = 'date')
    #bp_df = pd.read_csv('./data/bp_index.csv', parse_dates = ['date'], index_col = 'date')
    #liquidity_df = pd.read_csv('./data/liquidity_index.csv', parse_dates = ['date'], index_col = 'date')

    #print stock_industrycode_df
    factor_df = pd.read_csv('./data/factor_index.csv', index_col = ['date'], parse_dates = ['date'])
    cols = ['beta', 'market_value', 'momentum', 'dastd', 'bp', 'liquidity']
    factor_dfr = factor_df[cols]
    industry_index_dfr = industry_index_df.pct_change().fillna(0.0)
    stock_dfr = stock_df.pct_change().fillna(0.0)
    market_value_dfr = market_value_df.pct_change().fillna(0.0)
    beta_dfr = beta_df.pct_change().fillna(0.0)
    index_dfr = index_df.pct_change().fillna(0.0)['000905']

    dates = factor_dfr.index
    dates = dates[-750:]
    #print dates
    factor_dfr = factor_dfr.loc[dates]
    industry_index_dfr = industry_index_dfr.loc[dates]
    stock_dfr = stock_dfr.loc[dates]
    market_value_dfr = market_value_dfr.loc[dates]
    #print industry_index_dfr.columns
    #print industry_index_dfr

    for d in dates:
        tmp_stock_dfr = stock_dfr.loc[d]
        y = []
        X = []
        for code in tmp_stock_dfr.index:
            stock_code = code
            stock_r =  tmp_stock_dfr.loc[code]
            index_code = stock_industrycode_df.loc[stock_code, 'INDEXSYMBOL']
            industry_index_r = industry_index_dfr.loc[d, index_code]
            index_r = index_dfr.loc[d]
            market_value_r = market_value_dfr.loc[d, stock_code]
            beta_r = beta_dfr.loc[d, stock_code]
            if stock_r == 0.0:
                continue
            #print stock_r, industry_index_r, market_value_r
            y.append(stock_r)
            #X.append([industry_index_r, market_value_r])
            #X.append([industry_index_r, beta_r])
            #X.append([industry_index_r])
            X.append([index_r, beta_r])
            #print d, stock_r, index_r

        X = sm.add_constant(X)
        model  = sm.OLS(y,X)
        result = model.fit()
        print d, result.params
    #print factor_dfr
    #print industry_index_dfr
