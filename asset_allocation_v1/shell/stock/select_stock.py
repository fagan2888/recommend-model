#coding=utf8


import pandas as pd
import numpy  as np
import random


if __name__ == '__main__':

    momentum_df     = pd.read_csv('./tmp/momentum.csv', index_col = 'date', parse_dates = ['date'])
    market_value_df = pd.read_csv('./tmp/market_value.csv', index_col = 'date', parse_dates = ['date'])
    market_value_df = market_value_df.loc[momentum_df.index]

    dates = market_value_df.index
    d     = dates[-1]

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


    codes = market_value_codes & momentum_codes
    #codes = market_value_codes
    codes = list(codes)
    final_codes = set()
    while len(final_codes) < 5:
        n = random.randint(0, len(codes) - 1)
        final_codes.add(codes[n])
    #print final_codes
    #codes = final_codes

    print d,
    print codes
    print final_codes
