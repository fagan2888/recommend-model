#coding=utf8


import pandas as pd
import numpy as np
import statsmodels.api as sm



if __name__ == '__main__':

	macro_economic_df = pd.read_csv('./data/macro_economic.csv', index_col = ['date'], parse_dates = ['date'])
	#print macro_economic_df
	fiscal_policy_df = pd.read_csv('./data/fiscal_policy.csv', index_col = ['date'], parse_dates = ['date'])

	macro_inflation_df = pd.read_csv('./data/macro_inflation.csv', index_col = ['date'], parse_dates = ['date'])

	macro_interestrate_df = pd.read_csv('./data/macro_interestrate.csv', index_col = ['date'], parse_dates = ['date'])

	monetrary_supply_df = pd.read_csv('./data/monetrary_supply.csv', index_col = ['date'], parse_dates = ['date'])

	monetrary_foreign_df = pd.read_csv('./data/monetrary_foreign.csv', index_col = ['date'], parse_dates = ['date'])

	macro_index_df = pd.read_csv('./data/macro_index.csv', index_col = ['date'], parse_dates = ['date'])

	#print macro_index_df
	#print monetrary_foreign_df
	#print macro_inflation_df

	town_unemployment_rate = macro_economic_df['town_unemployment_rate'] / 100.0
	zz1000 =  macro_index_df['000852.SH'].fillna(method = 'pad')

	town_unemployment_rate = town_unemployment_rate.dropna()
	zz1000 = zz1000.dropna()
	zz1000 = zz1000.resample('M', how = 'last')
	town_unemployment_rate = town_unemployment_rate.reindex(zz1000.index)
	town_unemployment_rate = town_unemployment_rate.shift(1)
	town_unemployment_rate = town_unemployment_rate.dropna()

	index = town_unemployment_rate.index & zz1000.index
	town_unemployment_rate = town_unemployment_rate.loc[index]
	zz1000 = zz1000.loc[index]


	zz1000 = zz1000.shift(-1).dropna()
	zz1000_r = zz1000.pct_change().dropna()
	town_unemployment_rate_r = town_unemployment_rate.pct_change().fillna(0.0).loc[zz1000_r.index]

	#print zz1000_r
	#print town_unemployment_rate_r

	#print zz1000
	#print zz1000_r

	X = town_unemployment_rate_r.values
	y = zz1000_r.values
	#print np.corrcoef(X, y)

	X = sm.add_constant(X)

	model = sm.OLS(y, X)
	result = model.fit()
	print result.summary()
