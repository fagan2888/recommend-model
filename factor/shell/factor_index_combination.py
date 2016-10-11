#coding=utf8


import pandas as pd

factors = ['beta','market_value', 'momentum', 'dastd', 'bp', 'liquidity']
cols    = ['high_beta', 'low_beta', 'high_market_value', 'low_market_value', 'high_momentum', 'low_momentum', 'high_dastd', 'low_dastd', 'high_bp', 'low_bp', 'high_liquidity', 'low_liquidity']

dfs = []
for f in factors:
    path = './tmp/' + f + '_index.csv'
    df = pd.read_csv(path, index_col = 'date', parse_dates = ['date'])
    dfs.append(df)

df = pd.concat(dfs, axis = 1)
df.columns = cols
#df.fillna(0.0, inplace = True)

df.to_csv('./tmp/factor_index.csv')
