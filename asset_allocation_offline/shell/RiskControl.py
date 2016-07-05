#coding=utf8


import string
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import Portfolio as PF
import AllocationData
import FundIndicator
import numpy as np 


#drawdown_ratio = 0.6



def risk_control(df):


	dfr = df.pct_change().fillna(0.0)
	col = dfr.columns[0]

	dates = dfr.index

	risk_position      = 1
	risk_index         = 0
	add_position_index = 0

	day_returns = [0]

	all_drawdown   = [0]

	portfolio_vs    = [1]
	portfolio_dates = [dates[0]]

	risk_control_position = [1]


	for i in range(5, len(dates)):

		d       = dates[i]
		last_pv = portfolio_vs[-1]
		pv      = last_pv +  last_pv * dfr.loc[d, col] * risk_position
		portfolio_vs.append(pv)
		portfolio_dates.append(d)
		risk_control_position.append(risk_position)

		#print d, pv, risk_position

		r = 0 
		for n in range(0, 5):
			r = r + dfr.loc[dates[i - n], col]
		r = r / 5.0

		rs = []
		for n in range(0, 5):
			rs.append(dfr.loc[dates[i -n], col])
		print d, np.mean(rs), np.std(rs)		

		#r = dfr.loc[d, col]
		day_returns.sort()

		#print day_returns[(int)(0.1 * len(day_returns))]

		if r <= day_returns[(int)(0.1 * len(day_returns))] and r < 0.0:
			risk_position = risk_position * 0.4
			#if r <= day_returns[(int)(0.05 * len(day_returns))]:
			#	risk_position = risk_position * 0.0
			risk_index    = i
		elif i - risk_index >= 5 and i - add_position_index >= 5:
			risk_position = risk_position * 1.5
			if risk_position < 0.4:
				risk_position = 0.4
			if risk_position >= 0.9:
				risk_position = 1.0	 	
			add_position_index = i

		day_returns.append(r)


		'''
		drawdown = 1.0 -  pv / max(portfolio_vs)
		all_drawdown.sort()
		drawdown_threshold = all_drawdown[(int)(drawdown_ratio * len(all_drawdown))]
		all_drawdown.append(drawdown)


		if drawdown >= drawdown_threshold and r < 0:				
			risk_position = risk_position * 0.6
			risk_index    = i
		else:
			if i - risk_index >= 10:
				risk_position = risk_position * 1.5
				if risk_position < 0.4:
					risk_position = 0.4
				if risk_position > 1.0:
					risk_position = 1.0
		'''

	#print risk_control_position
	#print portfolio_vs


	asset_df = pd.DataFrame(portfolio_vs, index = portfolio_dates, columns = ['nav'])	
	asset_df.index.name = 'date'
	risk_position_df = pd.DataFrame(risk_control_position, index = portfolio_dates, columns = ['position'])
	risk_position_df.index.name = 'date'


	return asset_df ,risk_position_df 


if __name__ == '__main__':


	df = pd.read_csv('./tmp/asset.csv', index_col = 'date', parse_dates = 'date')

	asset_df, risk_position_df = risk_control(df)

	print asset_df 

	print "sharpe : ", FundIndicator.portfolio_sharpe_day(asset_df['nav'].values)
	print "annual_return : ", FundIndicator.portfolio_return_day(asset_df['nav'].values)
	print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(asset_df['nav'].values)

	asset_df.to_csv('./tmp/risk_asset.csv')
	risk_position_df.to_csv('./tmp/risk_position.csv')

	#print risk_position_df

