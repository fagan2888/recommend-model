#coding=utf8


import pandas as pd


#去除缺数据太多的基金
def clean(df):
	final_col = []
	for col in df.columns:
		col_df = df[col]
		length = 1.0 * len(col_df)
		col_df = col_df.dropna()
		if len(col_df) <= 0.95 * length:
			continue
		final_col.append(col)
	df = df[final_col]
	df = df.fillna(method='pad').fillna(method='bfill').dropna()
	return df


if __name__ == '__main__':
	df = pd.read_csv('./data/funds.csv', index_col='date', parse_dates=['date'])
	df = clean(df)
	df.to_csv('./data/clean_funds.csv')
