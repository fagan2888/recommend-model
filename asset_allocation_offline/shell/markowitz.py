#coding=utf8



import string
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import Portfolio as PF
import AllocationData



if __name__ == '__main__':


	#f    = './data/stock_bond.csv'
	f    = './data/index.csv'
	df   = pd.read_csv(f, index_col = 'date', parse_dates = ['date'])
	df   = df.resample('W-FRI').last()
	df   = df.fillna(method = 'pad')

	for col in df.columns:
		print df[col].replace([0.0], method = 'pad')

	#print df
	#df.to_csv('week.csv')
	dfr  = df.pct_change().fillna(0.0)

	dates= dfr.index
	dates= dates[400:-1]

	position_dates = []
	position_datas = []

	his_back = 26

	for i in range(his_back, len(dates)):

		start_date = dates[i- his_back]
		end_date   = dates[i]

		allocation_dfr = dfr[dfr.index <= end_date]
		allocation_dfr = allocation_dfr[allocation_dfr.index >= start_date]


		#print allocation_dfr
		risk, returns, ws, sharpe = PF.markowitz_r(allocation_dfr, None)
		fund_codes = allocation_dfr.columns

		print end_date.date(),
		for i in range(0, len(ws)):
			print ws[i],
		print 				

		position_dates.append(end_date)
		position_datas.append(ws)

	p_df = pd.DataFrame(position_datas, index = position_dates, columns = dfr.columns)
	
	#print p_df
