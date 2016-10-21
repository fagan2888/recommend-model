#coding=utf8


import pandas as pd
import numpy  as np
import os
import sys
sys.path.append('shell')
import datetime
import DBData
import DFUtil
import AllocationData
import re
from Const import datapath

pattern = re.compile(r'\d+')

def codes_r(d, dfr, codes):
	cs = pattern.findall(codes)
	r = 0.0
	for code in cs:
                if code in dfr.columns:
                        r = r + dfr.loc[d, code] / len(cs)
                        
	return r


def week2day(startdate, enddate):
	stock_df = pd.read_csv(datapath('stock_fund.csv'), index_col = 'date', parse_dates = ['date'])
	bond_df  = pd.read_csv(datapath('bond_fund.csv'),  index_col = 'date', parse_dates = ['date'])
	money_df = pd.read_csv(datapath('money_fund.csv'), index_col = 'date', parse_dates = ['date'], dtype={'money':str})


        if startdate == enddate:
                index_tmp = pd.DatetimeIndex([startdate])
        else:
                index_tmp = pd.DatetimeIndex([startdate, enddate])
        index = stock_df.index.union(bond_df.index).union(money_df.index).union(index_tmp)
        
	df = pd.concat([stock_df, bond_df, money_df], axis = 1, join_axes = [index]).fillna(method='pad')

        index = index[(index <= enddate)]

	rs      = []
	r_dates = []

	for start_date, end_date in zip(index[:-1], index[1:]):

		stock_value_df   = DBData.stock_day_fund_value(start_date, end_date)
		bond_value_df    = DBData.bond_day_fund_value(start_date, end_date)
		money_value_df   = DBData.money_day_fund_value(start_date, end_date)
                print start_date, end_date
		index_value_df   = DBData.index_day_value(start_date, end_date)

                if 'GLNC' not in index_value_df.columns:
                        index_value_df['GLNC'] = 287.8400


                index2 = stock_value_df.index
		stock_value_dfr  = stock_value_df.pct_change().fillna(0.0)[1:]
		bond_value_dfr   = bond_value_df.pct_change().reindex(index2).fillna(0.0)[1:]
		money_value_dfr  = money_value_df.pct_change().reindex(index2).fillna(0.0)[1:]
		index_value_dfr  = index_value_df.pct_change().reindex(index2).fillna(0.0)[1:]


		large_code       = df.loc[start_date, 'largecap']	
		small_code       = df.loc[start_date, 'smallcap']	
		rise_code        = df.loc[start_date, 'rise']	
		oscillation_code = df.loc[start_date, 'oscillation']	
		decline_code     = df.loc[start_date, 'decline']	
		growth_code      = df.loc[start_date, 'growth']	
		value_code       = df.loc[start_date, 'value']	

		ratebond_code    = df.loc[start_date,'ratebond']
		creditbond_code  = df.loc[start_date,'creditbond']
		convertiblebond_code = df.loc[start_date,'convertiblebond']

		money_code       = df.loc[start_date, 'money']
		#money_code       = "%06d" % df.loc[start_date, 'money']


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

			r.append(codes_r(tmp_d, money_value_dfr, money_code))

			r.append(index_value_dfr.loc[tmp_d, sp500_code])
			r.append(index_value_dfr.loc[tmp_d, gold_code])
			r.append(index_value_dfr.loc[tmp_d, hs_code])

			rs.append(r)

			tmp_d = datetime.datetime.strftime(tmp_d, '%Y-%m-%d')
			r_dates.append(datetime.datetime.strptime(tmp_d, '%Y-%m-%d'))
			print tmp_d ,r


	df = pd.DataFrame(rs, index = r_dates, columns = ['largecap','smallcap','rise', 'oscillation', 'decline', 'growth','value', 'ratebond', 'creditbond','convertiblebond','money', 'SP500.SPI', 'GLNC','HSCI.HI'])
	df.index.name = 'date'


	dfr = df
	values = []
	for col in dfr.columns:
		rs = dfr[col].values
		vs = [1]
		for i in range(1, len(rs)):
			r = rs[i]
			v = vs[-1] * ( 1.0 + r )	
			vs.append(v)
		values.append(vs)
	
	alldf = pd.DataFrame(np.matrix(values).T, index = dfr.index, columns = dfr.columns)

	alldf.to_csv(datapath('labelasset.csv'))

	week_df = alldf.resample('W-FRI').last()
	week_df = week_df.fillna(method = 'pad')
	week_df.to_csv(datapath('labelassetweek.csv'))


if __name__ == '__main__':
	print
