#coding=utf8


import string
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import Portfolio as PF
import AllocationData



def allocation_ratio(dfr):


	his_week = 15
	interval = 15

	position_dates = []
	position_datas = []

	dates        = dfr.index

	#result_dates.append(dates[his_week])

	fund_values  = {}
	fund_codes   = []


	for i in range(his_week, len(dates)):

		if (i - his_week) % interval == 0:

			start_date = dates[i- his_week].strftime('%Y-%m-%d')
			end_date   = dates[i].strftime('%Y-%m-%d')

			allocation_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d')]
			allocation_dfr = allocation_dfr[allocation_dfr.index >= datetime.strptime(start_date, '%Y-%m-%d')]

	
			#down_limit = []
			#up_limit   = []
			#for code in allocation_dfr.columns:
			#	down_limit.append(0.0)
			#	up_limit.append(0.8)	 	
			#risk, returns, ws, sharpe = PF.markowitz_r(allocation_dfr, [down_limit, up_limit])

			risk, returns, ws, sharpe = PF.markowitz_r(allocation_dfr, None)
			fund_codes = allocation_dfr.columns


			position_dates.append(dates[i])
			position_datas.append(ws)


	position_df = pd.DataFrame(position_datas, index=position_dates, columns=dfr.columns)
	position_df.index.name = 'date'

	return position_df



def allocation_asset(ratio_df, dfr):


	ratio_dates      = ratio_df.index
	start_date       = ratio_dates[0]

	dfr               = dfr[dfr.index >= start_date]
	dates             = dfr.index

	portfolio_vs = [1]

	assetlabels   = dfr.columns	
	asset_values = {}

	#print ratio_dates
	#print dates

	for asset in assetlabels:
		asset_values.setdefault(asset, [0.0])

	ratio_index = 0
	for i in range(0, len(dates)):

		d = dates[i]
		pv = 0
		for asset in assetlabels:
			vs = asset_values[asset]
			v  = vs[-1] * (1 + dfr.loc[d, asset])
			vs.append(v)	
			pv = pv + v

		if pv != 0:
			portfolio_vs.append(pv)
		
	
		#if d >= ratio_dates[ratio_index] and ratio_index < len(ratio_dates):
		if ratio_index < len(ratio_dates) and d >= ratio_dates[ratio_index]:
			for asset in assetlabels:
				pv = portfolio_vs[-1]
				asset_values[asset].append(pv * ratio_df.loc[ratio_dates[ratio_index], asset])
			ratio_index = ratio_index + 1


	asset_df = pd.DataFrame(portfolio_vs, index=dates, columns=['nav'])
	asset_df.index.name = 'date'


	return asset_df	




if __name__ == '__main__':


	stock_num = 4		
	bond_num  = 2
	df = pd.read_csv('./data/funds.csv', index_col = 'date', parse_dates = 'date')

	#dfr = df.pct_change().fillna(0.0)
	#week_df  = df.resample('W-FRI').last()
	#week_dfr = week_df.pct_change().fillna(0.0)


	df_equal_risk = pd.read_csv('./tmp/equalriskasset.csv', index_col = 'date', parse_dates = 'date')

	df = pd.concat( [ df_equal_risk[df_equal_risk.columns[0:stock_num]], df[df.columns[stock_num: stock_num + bond_num]] ], axis = 1, join_axes=[df_equal_risk.index])


	#df = df[ df.columns[stock_num:stock_num + bond_num] ]
	df = df[ df.columns[0 : stock_num] ]

	dfr = df.pct_change().fillna(0.0)
	#print df	

	week_df  = df.resample('W-FRI').last()
	week_dfr = week_df.pct_change().fillna(0.0)


	his_week = 15
	interval = 15


	position_df = allocation_ratio(week_dfr, his_week, interval)	
	asset_df    = allocation_asset(position_df, dfr)


	position_df.to_csv('./tmp/position.csv')
	asset_df.to_csv('./tmp/asset.csv')


	print asset_df
