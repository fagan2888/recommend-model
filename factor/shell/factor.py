#coding=utf8


import pandas as pd
import string
import statsmodels.api as sm
import numpy as np


#beta factor
def beta(stock_dfr, index_dfr):


    back = 252
    dates = stock_dfr.index

    betas = []
    for i in range(back, len(dates)):
        tmp_stock_dfr = stock_dfr.iloc[i - back: i,]
        tmp_index_dfr = index_dfr.loc[tmp_stock_dfr.index]
        tmp_stock_dfr.to_csv('tmp.csv')
        d = dates[i]
        beta = []
        for col in tmp_stock_dfr.columns:
            X = tmp_stock_dfr[col].values
            X = sm.add_constant(X)
            y = tmp_index_dfr.values
            try:
                model  = sm.OLS(y,X)
                result = model.fit()
                #print result.params
                beta.append(result.params[1])
            except:
                beta.append(0.0)
        betas.append(beta)
        print d


    beta_df = pd.DataFrame(betas, index = dates[252:], columns = stock_dfr.columns)
    #print beta_df
    beta_df.to_csv('./tmp/beta.csv')

    return beta_df


#momentum factor
def momentum(stock_dfr):
    T = 504
    L = 21
    dfr = pd.rolling_sum(stock_dfr, L)
    print dfr




if __name__ == '__main__':

    stock_df = pd.read_csv('./data/stock_price_adjust.csv', index_col = 'date', parse_dates = ['date'])
    #stock_df = pd.read_csv('./data/index_price.csv', index_col = 'date', parse_dates = ['date'])
    index_df = pd.read_csv('./data/index_price.csv', index_col = 'date', parse_dates = ['date'])
    index_df = index_df[['000905']]
    stock_dfr = stock_df.pct_change().fillna(0.0)
    index_dfr = index_df.pct_change().fillna(0.0)

    #beta(stock_dfr, index_dfr)
    momentum(stock_dfr)
