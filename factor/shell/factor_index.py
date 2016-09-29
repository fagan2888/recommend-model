#coding=utf8


import pandas as pd
import numpy  as np
import time


def index(factor_df, stock_dfr):

    factor_df = factor_df[stock_dfr.columns]

    factor_dates = factor_df.index.values
    stock_dates  = stock_dfr.index.values
    dates        = list(set(factor_dates) & set(stock_dates))
    dates.sort()
    codes = factor_df.columns

    rs = []
    ds = []
    for i in range(0, len(dates) - 1):
        d      = dates[i]
        next_d = dates[i + 1]

        factors         = factor_df.loc[d]
        factors.sort()
        l               = len(factors)
        factors         = factors[(int)(0.1 * l) : (int)(0.9 * l)]
        l               = len(factors)
        low_factors     = factors[0 :(int)(0.2 * l)]
        high_factors    = factors[(int)(0.2 * l) : -1]

        low_factors_rs  = []
        high_factors_rs = []
        for code in low_factors.index:
            low_factors_rs.append(stock_dfr.loc[next_d, code])
        for code in high_factors.index:
            high_factors_rs.append(stock_dfr.loc[next_d, code])
        r = (np.mean(high_factors_rs) - np.mean(low_factors_rs))

        print next_d, r
        rs.append(r)
        ds.append(next_d)

    df = pd.DataFrame(rs, index = ds, columns = ['r'])
    df.index.name = 'date'
    #print df
    return df


if __name__ == '__main__':

    beta_df = pd.read_csv('./tmp/beta.csv', index_col = 'date', parse_dates = ['date'])
    momentum_df = pd.read_csv('./tmp/momentum.csv', index_col = 'date', parse_dates = ['date'])
    market_value_df = pd.read_csv('./tmp/market_value.csv', index_col = 'date', parse_dates = ['date'])
    dastd_df        = pd.read_csv('./tmp/stock_dastd.csv', index_col = 'date', parse_dates = ['date'])

    stock_df = pd.read_csv('./data/stock_price_adjust.csv', index_col = 'date', parse_dates = ['date'])
    stock_dfr = stock_df.pct_change().fillna(0.0)


    #df = index(beta_df, stock_dfr)
    #df.to_csv('./tmp/beta_index.csv')

    #df = index(momentum_df, stock_dfr)
    #df.to_csv('./tmp/momentum_index.csv')

    #df = index(market_value_df, stock_dfr)
    #df.to_csv('./tmp/market_value_index.csv')

    df = index(dastd_df, stock_dfr)
    df.to_csv('./tmp/dastd_index.csv')
