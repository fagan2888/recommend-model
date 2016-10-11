#coding=utf8


import pandas as pd
import numpy  as np
import time


def trading_halt(factor_df, stock_df):

    dates = factor_df.index
    factor_df = factor_df[stock_df.columns]
    stock_dfr = stock_df.pct_change()
    codes = factor_df.columns
    factor_dates = factor_df.index.values
    stock_dates  = stock_dfr.index.values
    dates        = list(set(factor_dates) & set(stock_dates))
    dates.sort()

    for i in range(1, len(dates) - 1):
        d     = dates[i]
        pre_d = dates[i - 1]
        for code in codes:
            v = stock_df.loc[d, code]
            r = stock_dfr.loc[d, code]
            if np.isnan(v) or r > 9.98 / 100:
                factor_df.loc[d, code] = np.nan

        print 'trading halt', d

    return factor_df


def index(factor_df, stock_df):

    #factor_df = trading_halt(factor_df, stock_df)

    stock_dfr       = stock_df.pct_change().fillna(0.0)
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
        factors = factors.replace(-np.inf, np.nan)
        factors = factors.dropna()
        #print factors
        l               = len(factors)
        factors         = factors[(int)(0.1 * l) : (int)(0.9 * l)]
        l               = len(factors)
        low_factors     = factors[0 :(int)(0.2 * l)]
        high_factors    = factors[-1 * (int)(0.2 * l) : -1]

        low_factors_rs  = []
        high_factors_rs = []
        for code in low_factors.index:
            low_factors_rs.append(stock_dfr.loc[next_d, code])
        for code in high_factors.index:
            high_factors_rs.append(stock_dfr.loc[next_d, code])
        high_r= np.mean(high_factors_rs)
        low_r = np.mean(low_factors_rs)

        print next_d, high_r, low_r
        rs.append([high_r, low_r])
        ds.append(next_d)

    df = pd.DataFrame(rs, index = ds, columns = ['high_r','low_r']).fillna(0.0)
    df.index.name = 'date'
    #print df
    return df


if __name__ == '__main__':

    beta_df         = pd.read_csv('./tmp/beta.csv', index_col = 'date', parse_dates = ['date'])
    momentum_df     = pd.read_csv('./tmp/momentum.csv', index_col = 'date', parse_dates = ['date'])
    market_value_df = pd.read_csv('./tmp/market_value.csv', index_col = 'date', parse_dates = ['date'])
    dastd_df        = pd.read_csv('./tmp/stock_dastd.csv', index_col = 'date', parse_dates = ['date'])
    bp_df           = pd.read_csv('./tmp/bp.csv', index_col = 'date', parse_dates = ['date'])
    liquidity_df    = pd.read_csv('./tmp/liquidity.csv', index_col = 'date', parse_dates = ['date'])

    stock_df        = pd.read_csv('./data/stock_price_adjust.csv', index_col = 'date', parse_dates = ['date'])


    dfs = []

    #df = index(beta_df, stock_df)
    #df.to_csv('./tmp/beta_index.csv')
    #dfs.append(df)

    #df = index(momentum_df, stock_df)
    #df.to_csv('./tmp/momentum_index.csv')
    #dfs.append(df)

    #df = index(market_value_df, stock_df)
    #df.to_csv('./tmp/market_value_index.csv')
    #dfs.append(df)

    #df = index(dastd_df, stock_df)
    #df.to_csv('./tmp/dastd_index.csv')
    #dfs.append(df)

    #df = index(bp_df, stock_df)
    #df.to_csv('./tmp/bp_index.csv')
    #dfs.append(df)

    df = index(liquidity_df, stock_df)
    df.to_csv('./tmp/liquidity_index.csv')
    #dfs.append(df)
