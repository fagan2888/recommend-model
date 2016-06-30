#coding=utf8


import pandas as pd


def combine():

	hdf = pd.read_csv('./tmp/highriskposition.csv', index_col = 'date', parse_dates = 'date')
	ldf = pd.read_csv('./tmp/lowriskposition.csv', index_col = 'date', parse_dates = 'date')
	hldf = pd.read_csv('./tmp/highlowriskposition.csv', index_col = 'date', parse_dates = 'date')

	hdfc = hdf.columns
	ldfc = ldf.columns
	hldfc = hldf.columns
	dates = hldf.index
	hdates = hdf.index
	ldates = ldf.index
	result = []
	all_columns = []
	dvalue = []
	for date in dates:
		dvalue.append(date)
		high_pos = hldf.loc[date, 'high_risk_asset']
		low_pos = hldf.loc[date, 'low_risk_asset']
		tmp_arr = []
		hpredate = hdates[0]
		for hdate in hdates:
			if hdate > date:
				print hdate, date
				for col in hdfc:
					col_value_h = hdf.loc[hpredate, col] * high_pos
					tmp_arr.append(col_value_h)
				break
			elif hdate == date:
				print hdate, date
				for col in hdfc:
					col_value_h = hdf.loc[hdate, col] * high_pos
					tmp_arr.append(col_value_h)
				break
			hpredate = hdate
		lpredate = ldates[0]
		for ldate in ldates:
			if ldate > date:
				print ldate, date
				for col in ldfc:
					col_value_l = ldf.loc[lpredate, col] * low_pos
					tmp_arr.append(col_value_l)
				break
			elif ldate == date:
				print ldate, date
				for col in ldfc:
					col_value_l = ldf.loc[ldate, col] * low_pos
					tmp_arr.append(col_value_l)
				break
			lpredate = ldate
		if len(tmp_arr):
			result.append(tmp_arr)
	print result
	print len(result)
	if len(result) != len(dvalue):
		dvalue.pop(len(dvalue) - len(result))
	for col in hdfc:
		all_columns.append(col)
	for col in ldfc:
		all_columns.append(col)
	result_df = pd.DataFrame(result, index=dvalue, columns=all_columns)
	result_df.index.name = 'date'
	result_df.to_csv('./tmp/allposition.csv')


