#coding=utf8


import pandas as pd
import numpy as np
import statsmodels.api as sm


def resample_delay_process(factor, index_price, resample, factor_delay):

	index_price = index_price.resample(resample, how = 'last')
	factor = factor.resample(resample, how = 'last')
	factor = factor.shift(factor_delay)

	factor = factor.dropna()

	index = index_price.index & factor.index
	factor = factor.loc[index]
	index_price = index_price.loc[index]
	
	return factor, index_price


if __name__ == '__main__':

	macro_economic_df = pd.read_csv('./data/macro_economic.csv', index_col = ['date'], parse_dates = ['date'])
	#print macro_economic_df
	fiscal_policy_df = pd.read_csv('./data/fiscal_policy.csv', index_col = ['date'], parse_dates = ['date'])

	macro_inflation_df = pd.read_csv('./data/macro_inflation.csv', index_col = ['date'], parse_dates = ['date'])

	macro_interestrate_df = pd.read_csv('./data/macro_interestrate.csv', index_col = ['date'], parse_dates = ['date'])

	monetrary_supply_df = pd.read_csv('./data/monetrary_supply.csv', index_col = ['date'], parse_dates = ['date'])

	monetrary_foreign_df = pd.read_csv('./data/monetrary_foreign.csv', index_col = ['date'], parse_dates = ['date'])

	macro_index_df = pd.read_csv('./data/macro_index.csv', index_col = ['date'], parse_dates = ['date'])

	#print macro_index_df.columns
	#print monetrary_foreign_df
	#print macro_inflation_df
	#print macro_economic_df.columns
	col = macro_economic_df.columns[3]
	print macro_economic_df[col].dropna()


	'''
	town_unemployment_rate = macro_economic_df['town_unemployment_rate'] / 100.0
	zz1000 =  macro_index_df['000852.SH'].fillna(method = 'pad')

	town_unemployment_rate, zz1000 = resample_delay_process(town_unemployment_rate, zz1000, 'M', 1)
	#print town_unemployment_rate
	#print zz1000

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
	'''
