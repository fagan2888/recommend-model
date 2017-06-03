#coding=utf8


import pandas as pd
import numpy as np


if __name__ == '__main__':

	
	macro_economic_df = pd.read_csv('./data/macro_economic.csv', index_col = ['date'], parse_dates = ['date'])
	fiscal_policy_df = pd.read_csv('./data/fiscal_policy.csv', index_col = ['date'], parse_dates = ['date'])
	macro_inflation_df = pd.read_csv('./data/macro_inflation.csv', index_col = ['date'], parse_dates = ['date'])
	macro_interestrate_df = pd.read_csv('./data/macro_interestrate.csv', index_col = ['date'], parse_dates = ['date'])
	monetrary_supply_df = pd.read_csv('./data/monetrary_supply.csv', index_col = ['date'], parse_dates = ['date'])
	monetrary_foreign_df = pd.read_csv('./data/monetrary_foreign.csv', index_col = ['date'], parse_dates = ['date'])
	macro_index_df = pd.read_csv('./data/macro_index.csv', index_col = ['date'], parse_dates = ['date'])

	#交易日变成自然日
	macro_index_df = macro_index_df.fillna(method = 'pad')
	macro_index = macro_index_df.index
	start_date = macro_index[0]
	end_date = macro_index[-1]
	dates = pd.date_range(start_date, end_date)
	macro_index_df = macro_index_df.reindex(dates)
	macro_index_df = macro_index_df.fillna(method = 'pad')

	#数据合并		
	df = pd.concat([macro_economic_df, macro_interestrate_df, macro_inflation_df, monetrary_supply_df, monetrary_foreign_df, fiscal_policy_df, macro_index_df], axis = 1, join_axes = [macro_index_df.index])

	#按月resample
	df = df.resample('M').last()

	#处理季度累计数据
	tmp_df = df['urban_income'].copy()
	tmp_df = tmp_df.dropna()
	tmp_df = tmp_df.groupby(by = tmp_df.index.year).apply(lambda x : x.rolling(window = 2, min_periods = 1).apply(lambda x : x[1] - x[0] if len(x) >= 2 else x[0]))
	tmp_df = tmp_df.reindex(df.index)
	df['urban_income'] = tmp_df

	#做线性插值
	df['town_unemployment_rate'] = df['town_unemployment_rate'].interpolate()
	df['town_unemployment_insurance_num'] = df['town_unemployment_insurance_num'].interpolate()
	df['urban_income'] = df['urban_income'].interpolate()
	df['gdp_yoy'] = df['gdp_yoy'].interpolate()
	df['payment_balance_budget'] = df['payment_balance_budget'].interpolate()


	#更改同比计数方式
	df['cgpi_yoy'] = df['cgpi_yoy'] - 100

	#更改数据类型
	df.astype({'industrial_value_added_yoy' : float, 'payment_balance_budget' : float})
	
	'''
	计算延迟期数
	df = df.iloc[-4:]
	#print df
	for col in df.columns:
		print "'" + col + "':", 4 - len(df[col].dropna())
	'''

	#保存文件
	df.index.name = 'date'
	df.to_csv('./data/macro_factor_index.csv')