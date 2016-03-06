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
	indicator_df = pd.read_csv('./wind/fund_establish_date.csv', index_col = 'code', parse_dates = [1])
	establish_date_code = set()
	for code in indicator_df.index:
		date = indicator_df['establish_date'][code]		
		if date <= datetime.strptime(start_date, '%Y-%m-%d'):
			establish_date_code.add(code)


	cols = df.columns
	fund_cols = []
	for col in cols:

		#有20%的净值是nan，则过滤掉该基金
		vs = df[col]	
		n = 0
		for v in vs:
			if isnan(v):
				n = n + 1	
		if n > 0.2 * len(vs):
			continue

		if col.find('OF') >= 0 and col in establish_date_code:
			fund_cols.append(col)



	fund_df = df[fund_cols]
	#fund_df_r = df_r[fund_cols]


	index_df = df[index_code]
	#index_df_r = df_r[index_col]	 

	#print fund_df_r
	#print index_df_r

	return fund_df, index_df



def fund_value(start_date, end_date):

	
	#取开始时间和结束时间的数据	
	df = pd.read_csv('./wind/fund_value.csv', index_col = 'date', parse_dates = [0])
	df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
	df = df[ df.index >= datetime.strptime(start_date,'%Y-%m-%d')]

	#取基金成立时间指标
	indicator_df = pd.read_csv('./wind/fund_establish_date.csv', index_col = 'code', parse_dates = [1])
	establish_date_code = set()
	for code in indicator_df.index:
		date = indicator_df['establish_date'][code]		
		if date <= datetime.strptime(start_date, '%Y-%m-%d'):
			establish_date_code.add(code)


	cols = df.columns
	fund_cols = []
	for col in cols:

		#有20%的净值是nan，则过滤掉该基金
		vs = df[col].values	
		n = 0
		for v in vs:
			if isnan(v):
				n = n + 1	
		if n > 0.2 * len(vs):
			continue

		if col.find('OF') >= 0 and col in establish_date_code:
			fund_cols.append(col)


	fund_df = df[fund_cols]

	return fund_df



def index_value(start_date, end_date, index_code):

	#取开始时间和结束时间的数据	
	df = pd.read_csv('./wind/fund_value.csv', index_col = 'date', parse_dates = [0] )
	df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
	df = df[ df.index >= datetime.strptime(start_date,'%Y-%m-%d')]

	index_df = df[index_code]

	return index_df




def establish_data():
	indicator_df = pd.read_csv('./wind/fund_establish_date.csv', index_col = 'code', parse_dates = [1])	
	return indicator_df



def scale_data():
	indicator_df = pd.read_csv('./wind/fund_scale.csv', index_col = 'code')	
	return indicator_df



def stock_fund_code():
	funddf = pd.read_csv('./wind/stock_fund_code.csv', index_col = 'code')	
	codes = []
	for code in funddf.index:
		codes.append(code)
	return codes



def fund_position(start_date, end_date):

	positiondf = pd.read_csv('./wind/fund_position.csv', index_col = 'date' , parse_dates = [0])	
	positiondf = positiondf[ positiondf.index <= datetime.strptime(end_date,'%Y-%m-%d')]
	positiondf = positiondf[ positiondf.index >= datetime.strptime(start_date,'%Y-%m-%d')]

	codes = []

	for col in positiondf.columns:
		vs = positiondf[col].values
		has = True
		for v in vs:
			if isnan(v):
				has = False

		if has:
			codes.append(col)						


	positiondf = positiondf[codes]										
	return positiondf	


if __name__ == '__main__':
	
	fund_df, index_df = fund_index_data('2011-02-03', '2015-03-02', '000300.SH')
	#print fund_df, index_df 
	#print np.mean(index_df.pct_change())
	#fund_scale =  scale_data()
	#print fund_scale.values
	#print stock_fund_code()
	print fund_position('2011-01-02','2012-12-31')

