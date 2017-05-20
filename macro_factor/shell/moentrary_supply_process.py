#coding=utf8


import pandas as pd


if __name__ == '__main__':

	df = pd.read_csv('./data/monetrary_supply.csv', index_col = ['date'], parse_dates = ['date'])
	#print df.columns

	df['m1_yoy_m2_yoy'] = df['m1_yoy'] - df['m2_yoy']
	print df
	df.to_csv('monetrary_supply.csv')
