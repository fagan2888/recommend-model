#coding=utf8


import pandas as pd
import sys
sys.path.append('shell')
import config
import DBData
import statsmodels.api as sm
import numpy as np


if __name__ == '__main__':


    start_date = '2014-01-01'
    end_date   = '2016-10-30'


    factors = ['beta','market_value', 'momentum', 'dastd', 'bp', 'liquidity']
    cols    = ['high_beta', 'low_beta', 'high_market_value', 'low_market_value', 'high_momentum', 'low_momentum', 'high_dastd', 'low_dastd', 'high_bp', 'low_bp', 'high_liquidity', 'low_liquidity']

    style_df = pd.read_csv('./data/factor_index.csv', index_col = ['date'], parse_dates = ['date'])
    style_df = style_df.fillna(0.0)
    style_df = style_df[cols]

    fund_df = pd.read_csv('./data/industry_test_fund.csv', index_col = ['date'], parse_dates = ['date'])
    df = fund_df[['519983']]
    dfr = df.pct_change().fillna(0.0)


    dates = dfr.index & style_df.index
    back = 252
    style_df = style_df.loc[dates]
    dfr = dfr.loc[dates]
    #print industry_dfr.index
    #print dfr.index

    results = []
    ds = []
    total = 0
    correct = 0
    bias = 0
    for i in range(back, len(dates) - 1):
        d = dates[i]
        style_r = style_df.iloc[i - back : i]
        #print industry_r.values
        fund_r     = dfr.iloc[i - back : i]
        #print industry_r
        #print fund_r

        #print fund_r.values
        X = style_r.values
        X = sm.add_constant(X)
        y = fund_r.values
        model  = sm.OLS(y,X)
        result = model.fit()
        #print result.summary()
        #print result.params

        rsquared_adj = result.rsquared_adj
        style_r = np.append(np.array(1), style_df.iloc[i].values)
        fund_r = dfr.iloc[i].values
        coef = result.params
        real = fund_r
        predict = np.dot(style_r, coef)
        if real >= 0 and predict >= 0:
            correct += 1
        elif real <= 0 and predict <= 0:
            correct += 1
        total += 1
        bias += (predict - real) ** 2
        ds.append(d)
        results.append([predict, real[0], rsquared_adj])
        print d, predict, real[0], rsquared_adj
        #print d, coef
    result_df = pd.DataFrame(results, index = ds, columns = ['predict', 'real', 'r2'])
    #result_df.to_csv('style_factor_index.csv')
    print 1.0 * correct / total
    print (bias / total) ** 0.5
