#coding=utf8


import pandas as pd
import numpy  as np
import random


if __name__ == '__main__':

    momentum_df     = pd.read_csv('./tmp/momentum.csv', index_col = 'date', parse_dates = ['date'])
    market_value_df = pd.read_csv('./tmp/market_value.csv', index_col = 'date', parse_dates = ['date'])
    market_value_df = market_value_df.loc[momentum_df.index]
    stock_df        = pd.read_csv('./data/stock_price_adjust.csv', index_col = 'date', parse_dates = ['date'])
    stock_dfr       = stock_df.pct_change().fillna(0.0)


    dates = market_value_df.index
    #dates = dates[100 : ]
    #print dates
    #dates = dates[0 : ]
    vs = []
    ds = []
    codes = None
    for i in range(0 ,len(dates) - 2):
        d      = dates[i]
        next_d = dates[i + 1]

        market_value = market_value_df.loc[d]
        market_value.sort()
        market_value.replace(-np.inf, np.nan)
        market_value = market_value.dropna()
        l = len(market_value)
        market_value = market_value[(int)(0.05 * l) : (int)(0.1 * l)]

        momentum = momentum_df.loc[d]
        momentum.sort()
        momentum.replace(-np.inf, np.nan)
        momentum = momentum.dropna()
        l = len(momentum)
        momentum = momentum[(int)(0.05 * l) : (int)(0.15 * l)]

        market_value_codes = set()
        for code in market_value.index:
            market_value_codes.add(code)

        momentum_codes = set()
        for code in momentum.index:
            momentum_codes.add(code)


        if i % 21 == 0:
            codes = market_value_codes & momentum_codes
            if len(codes) == 0:
                continue
            #codes = market_value_codes
            #codes = momentum_codes
            #print codes
            #codes = stock_df.loc[d].dropna().index.values
            #print codes
            codes = list(codes)
            final_codes = set()
            while len(final_codes) < 6 and len(final_codes) < len(codes):
                n = random.randint(0, len(codes) - 1)
                final_codes.add(codes[n])
            #print final_codes
            codes = final_codes

        #print d
        num = len(codes)
        r  = 0
        for code in codes:
            #print code, stock_dfr.loc[next_d, code]
            r = r + stock_dfr.loc[next_d, code] / num
        if len(vs) == 0:
            vs = [1]
        else:
            vs.append(vs[-1] * ( 1 + r ))
        ds.append(next_d)
        #print
        print next_d, vs[-1]
    df = pd.DataFrame(vs, index = ds, columns = ['nav'])
    #print df
    df.to_csv('nav.csv')
