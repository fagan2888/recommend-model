#coding=utf8


import pandas as pd

factors = ['beta','market_value', 'momentum', 'dastd', 'bp', 'liquidity']
cols    = ['high_beta', 'low_beta', 'high_market_value', 'low_market_value', 'high_momentum', 'low_momentum', 'high_dastd', 'low_dastd', 'high_bp', 'low_bp', 'high_liquidity', 'low_liquidity']

dfs = []
#for f in factors:
#    path = './data/' + f + '_index.csv'
#    df = pd.read_csv(path, index_col = 'date', parse_dates = ['date'])
#    dfs.append(df)

df = pd.read_csv('./data/beta_index.csv', parse_dates = ['date'], index_col = 'date')
dfs.append(df)
df = pd.read_csv('./data/market_value_index.csv', parse_dates = ['date'], index_col = 'date')
dfs.append(df)
df = pd.read_csv('./data/momentum_index.csv', parse_dates = ['date'], index_col = 'date')
dfs.append(df)
df = pd.read_csv('./data/dastd_index.csv', parse_dates = ['date'], index_col = 'date')
dfs.append(df)
df = pd.read_csv('./data/bp_index.csv', parse_dates = ['date'], index_col = 'date')
dfs.append(df)
df = pd.read_csv('./data/liquidity_index.csv', parse_dates = ['date'], index_col = 'date')
dfs.append(df)

df = pd.concat(dfs, axis = 1)
df.columns = cols
df['beta']         = df['low_beta']         - df['high_beta']
df['market_value'] = df['low_market_value'] - df['high_market_value']
df['momentum']     = df['low_momentum']     - df['high_momentum']
df['dastd']        = df['low_dastd']        - df['high_dastd']
df['bp']           = df['low_bp']           - df['high_bp']
df['liquidity']    = df['low_liquidity']    - df['high_liquidity']
#df.fillna(0.0, inplace = True)

df.to_csv('./data/factor_index.csv')
print df
