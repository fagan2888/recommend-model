#coding=utf8

import pandas as pd

'''
ra_fund_pool_df = pd.read_csv('ra_pool_nav.csv', index_col = ['date'], parse_dates = ['date'])
factor_df = pd.read_csv('./tmp/factor_vs.csv', index_col = ['date'], parse_dates = ['date'])
ra_fund_pool_df = ra_fund_pool_df.loc[factor_df.index]
ra_fund_pool_df = ra_fund_pool_df / ra_fund_pool_df.iloc[0]

df = pd.concat([ra_fund_pool_df, factor_df], axis = 1)
df.columns = ['size_invshare_nav','factor_nav']
print df
df.to_csv('df.csv')
'''


risk_asset_allocation_df = pd.read_csv('./risk_asset_allocation_nav_.csv', index_col = ['date'], parse_dates = ['date'])
equal_df = pd.read_csv('./tmp/equalriskasset.csv', index_col = ['date'], parse_dates = ['date'])
equal_dfr = equal_df.pct_change().fillna(0.0)
equal_dfr = equal_dfr.iloc[0:-1,]
equal_dfr['r'] = equal_dfr.sum(axis = 1) / len(equal_dfr.columns)
#print equal_dfr.columns
equal_df = (1 + equal_dfr).cumprod()

df = pd.concat([risk_asset_allocation_df, equal_df], join_axes = [equal_df.index], axis = 1)
#print df
df.to_csv('nav.csv')
#print risk_asset_allocation_df
