#coding=utf8


import pandas as pd
import numpy  as np
import sys
sys.path.append('shell')
import datetime
import DBData
import DFUtil
import AllocationData
import re


pattern = re.compile(r'\d+')

def codes_r(d, dfr, codes):
	cs = pattern.findall(codes)
	r = 0.0
	for code in cs:
		r = r + dfr.loc[d, code] / len(cs)
	return r


def week2day(allocationdata):


	stock_df = pd.read_csv('./tmp/stock_fund.csv', index_col = 'date', parse_dates = ['date'])
	bond_df  = pd.read_csv('./tmp/bond_fund.csv',  index_col = 'date', parse_dates = ['date'])
	money_df = pd.read_csv('./tmp/money_fund.csv', index_col = 'date', parse_dates = ['date'])


	#print stock_df
	#print bond_df
	#print money_df
	#print other_df


	df = pd.concat([stock_df, bond_df, money_df], axis = 1, join_axes = [stock_df.index])
	dates = df.index

	rs      = []
	r_dates = []

	for i in range(0, len(dates)):


		d = dates[i]
		start_date = d
		end_date   = datetime.datetime.now()
		if i < len(dates) - 1:
			end_date = dates[i + 1]	

		stock_value_df   = DBData.stock_day_fund_value(start_date, end_date)
		bond_value_df    = DBData.bond_day_fund_value(start_date, end_date)
		money_value_df   = DBData.money_day_fund_value(start_date, end_date)
		index_value_df   = DBData.index_day_value(start_date, end_date)


		stock_value_dfr  = stock_value_df.pct_change().fillna(0.0).iloc[1:,:]
		bond_value_dfr   = bond_value_df.pct_change().fillna(0.0).iloc[1:,:]
		money_value_dfr  = money_value_df.pct_change().fillna(0.0).iloc[1:,:]
		index_value_dfr  = index_value_df.pct_change().fillna(0.0).iloc[1:,:]


		large_code       = stock_df.loc[d, 'largecap']	
		small_code       = stock_df.loc[d, 'smallcap']	
		rise_code        = stock_df.loc[d, 'rise']	
		oscillation_code = stock_df.loc[d, 'oscillation']	
		decline_code     = stock_df.loc[d, 'decline']	
		growth_code      = stock_df.loc[d, 'growth']	
		value_code       = stock_df.loc[d, 'value']	

		ratebond_code    = bond_df.loc[d,'ratebond']
		creditbond_code  = bond_df.loc[d,'creditbond']
		convertiblebond_code = bond_df.loc[d,'convertiblebond']

		#money_code       = money_df.loc[d, 'money']
		money_code       = "%06d" % money_df.loc[d, 'money']


		sp500_code       = 'SP500.SPI'
		gold_code        = 'GLNC'
		hs_code          = 'HSCI.HI'


		for tmp_d in stock_value_dfr.index:

			r = []

			r.append(codes_r(tmp_d, stock_value_dfr, large_code))
			r.append(codes_r(tmp_d, stock_value_dfr, small_code))
			r.append(codes_r(tmp_d, stock_value_dfr, rise_code))
			r.append(codes_r(tmp_d, stock_value_dfr, oscillation_code))
			r.append(codes_r(tmp_d, stock_value_dfr, decline_code))
			r.append(codes_r(tmp_d, stock_value_dfr, growth_code))
			r.append(codes_r(tmp_d, stock_value_dfr, value_code))

			r.append(codes_r(tmp_d, bond_value_dfr, ratebond_code))
			r.append(codes_r(tmp_d, bond_value_dfr, creditbond_code))
			r.append(codes_r(tmp_d, bond_value_dfr, convertiblebond_code))

			r.append(money_value_dfr.loc[tmp_d, money_code])

			r.append(index_value_dfr.loc[tmp_d, sp500_code])
			r.append(index_value_dfr.loc[tmp_d, gold_code])
			r.append(index_value_dfr.loc[tmp_d, hs_code])

			rs.append(r)

			tmp_d = datetime.datetime.strftime(tmp_d, '%Y-%m-%d')
			r_dates.append(datetime.datetime.strptime(tmp_d, '%Y-%m-%d'))
			print tmp_d ,r


	df = pd.DataFrame(rs, index = r_dates, columns = ['largecap','smallcap','rise', 'oscillation', 'decline', 'growth','value', 'ratebond', 'creditbond','convertiblebond','money', 'SP500.SPI', 'GLNC','HSCI.HI'])
	df.index.name = 'date'
	df.to_csv('./tmp/labelasset.csv')
	
	allocationdata.label_asset_df = df	
		#print start_date, d, end_date			
	#print df


if __name__ == '__main__':


	allocationdata = AllocationData.allocationdata()
	week2day(allocationdata)
