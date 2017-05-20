#coding=utf8


import pandas as pd


if __name__ == '__main__':

	macro_economic_df = pd.read_csv('./data/macro_economic.csv', index_col = ['date'], parse_dates = ['date'])
	cu_shf_df = pd.read_csv('./data/cu.shf.csv', index_col = ['date'], parse_dates = ['date'])
	#print df.columns

	macro_economic_df = pd.concat([macro_economic_df, cu_shf_df], axis = 1, join_axes = [macro_economic_df.index])
	print macro_economic_df


	#df['m1_yoy_m2_yoy'] = df['m1_yoy'] - df['m2_yoy']
	#print df
	macro_economic_df.to_csv('macro_economic.csv')
