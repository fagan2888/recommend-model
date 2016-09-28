#coding=utf8


import pandas as pd
import numpy  as np
import time


def index(beta_df, stock_dfr):

    dates = beta_df.index
    codes = beta_df.columns

    rs = []
    for d in dates:
        betas         = beta_df.loc[d]
        betas.sort()
        l             = len(betas)
        betas         = betas[(int)(0.1 * l) : (int)(0.9 * l)]
        l             = len(betas)
        low_betas     = betas[0 :(int)(0.2 * l)]
        high_betas    = betas[(int)(0.2 * l) : -1]
        low_betas_sum = np.sum(low_betas.values)
        high_betas_sum= np.sum(high_betas.values)

        low_betas_rs  = []
        high_betas_rs = []
        for code in low_betas.index:
            low_betas_rs.append(stock_dfr.loc[d, code])
        for code in high_betas.index:
            high_betas_rs.append(stock_dfr.loc[d, code])
        r = (np.mean(high_betas_rs) - np.mean(low_betas_rs))
        print d, r
        rs.append(r)

    df = pd.DataFrame(rs, index = dates, columns = ['r'])
    #print df
    return df


if __name__ == '__main__':

    beta_df = pd.read_csv('./tmp/beta.csv', index_col = 'date', parse_dates = ['date'])
    momentum_df = pd.read_csv('./tmp/momentum.csv', index_col = 'date', parse_dates = ['date'])

    stock_df = pd.read_csv('./data/stock_price_adjust.csv', index_col = 'date', parse_dates = ['date'])
    stock_dfr = stock_df.pct_change().fillna(0.0)

    #df = beta_index(beta_df, stock_dfr)
    #df.to_csv('./tmp/beta_index.csv')

    df = index(momentum_df, stock_dfr)
    df.to_csv('./tmp/momentum_index.csv')
    print df
