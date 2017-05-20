#coding=utf8


import pandas as pd


if __name__ == '__main__':

	df = pd.read_csv('./data/macro_interestrate.csv', index_col = ['date'], parse_dates = ['date'])
	#print df.columns

	df['10ytbr_3mtbr'] = df['10ytbr'] - df['3mtbr']
	df['10ytbr_2ytbr'] = df['10ytbr'] - df['2ytbr']
	print df
	df.to_csv('macro_interestrate.csv')
	
