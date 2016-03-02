#coding=utf8


import string
import pandas as pd
from datetime import datetime
from numpy import *
import numpy as np


#start_date = '2014-03-07'
#end_date = '2015-02-09'
#index_col = '000300.SH'

#基金和指数数据抽取和对齐

def fund_index_data(start_date, end_date, index_code):

	#取开始时间和结束时间的数据	
	df = pd.read_csv('./wind/fund_value.csv', index_col = 'date', parse_dates = [0] )
	df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
	df = df[ df.index >= datetime.strptime(start_date,'%Y-%m-%d')]


	#取基金成立时间指标
	indicator_df = pd.read_csv('./wind/fund_indicator.csv', index_col = 'code', parse_dates = [1])
	establish_date_code = set()
	for code in indicator_df.index:
		date = indicator_df['establish_date'][code]		
		if date <= datetime.strptime(start_date, '%Y-%m-%d'):
			establish_date_code.add(code)


	cols = df.columns
	fund_cols = []
	for col in cols:
		if col.find('OF') >= 0 and col in establish_date_code:
			fund_cols.append(col)



	fund_df = df[fund_cols]
	#fund_df_r = df_r[fund_cols]


	index_df = df[index_code]
	#index_df_r = df_r[index_col]	 

	#print fund_df_r
	#print index_df_r

	return fund_df, index_df




if __name__ == '__main__':
	
	fund_df, index_df = fund_index_data('2011-02-03', '2015-03-02', '000300.SH')
	print fund_df, index_df 
	print np.mean(index_df.pct_change())

