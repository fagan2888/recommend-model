#coding=utf8

import pandas as pd

ra_fund_pool_df = pd.read_csv('ra_pool_nav.csv', index_col = ['date'], parse_dates = ['date'])
factor_df = pd.read_csv('./tmp/factor_vs.csv', index_col = ['date'], parse_dates = ['date'])
ra_fund_pool_df = ra_fund_pool_df.loc[factor_df.index]
ra_fund_pool_df = ra_fund_pool_df / ra_fund_pool_df.iloc[0]

df = pd.concat([ra_fund_pool_df, factor_df], axis = 1)
df.columns = ['size_invshare_nav','factor_nav']
print df
df.to_csv('df.csv')
