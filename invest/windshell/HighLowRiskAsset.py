#coding=utf8


import string
import pandas as pd
import numpy as np
import fundindicator
from datetime import datetime
import portfolio as pf
import fundindicator as fi


def highriskasset(dfr, his_week, interval):


	result_dates = []
	result_datas = []

	position_dates = []
	position_datas = []

	dates        = dfr.index

	portfolio_vs = [1]
	result_dates.append(dates[his_week])

	fund_values  = {}
	fund_codes   = []

	for i in range(his_week + 1, len(dates)):


		if (i - his_week - 1 ) % interval == 0:

			start_date = dates[i- his_week].strftime('%Y-%m-%d')
			end_date   = dates[i - 1].strftime('%Y-%m-%d')

			allocation_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d')]
			allocation_dfr = allocation_dfr[allocation_dfr.index >= datetime.strptime(start_date, '%Y-%m-%d')]

			risk, returns, ws, sharpe = pf.markowitz_r(allocation_dfr, None)
			fund_codes = allocation_dfr.columns


			last_pv = portfolio_vs[-1]
			fund_values = {}
			for n in range(0, len(fund_codes)):
				fund_values[n] = [last_pv * ws[n]]

			position_dates.append(end_date)
			position_datas.append(ws)


		pv = 0
		d = dates[i]
		for n in range(0, len(fund_codes)):
			vs = fund_values[n]
			code = fund_codes[n]
			fund_last_v = vs[-1]
			fund_last_v = fund_last_v + fund_last_v * dfr.loc[d, code]
			vs.append(fund_last_v)
			pv = pv + vs[-1]
		portfolio_vs.append(pv)
		result_dates.append(d)

		print d , pv


	result_datas  = portfolio_vs
	result_df = pd.DataFrame(result_datas, index=result_dates,
								 columns=['high_risk_asset'])

	result_df.to_csv('./tmp/highriskasset.csv')


	highriskposition_df = pd.DataFrame(position_datas, index=position_dates, columns=dfr.columns)
	highriskposition_df.to_csv('./tmp/highriskposition.csv')

	return result_df




def lowriskasset(dfr, his_week, interval):


	result_dates = []
	result_datas = []


	position_dates = []
	position_datas = []


	dates        = dfr.index

	portfolio_vs = [1]
	result_dates.append(dates[his_week])

	fund_values  = {}
	fund_codes   = []

	for i in range(his_week + 1, len(dates)):


		if (i - his_week - 1 ) % interval == 0:

			start_date = dates[i- his_week].strftime('%Y-%m-%d')
			end_date   = dates[i - 1].strftime('%Y-%m-%d')

			allocation_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d')]
			allocation_dfr = allocation_dfr[allocation_dfr.index >= datetime.strptime(start_date, '%Y-%m-%d')]

			risk, returns, ws, sharpe = pf.markowitz_r(allocation_dfr, None)
			fund_codes = allocation_dfr.columns

			last_pv = portfolio_vs[-1]
			fund_values = {}
			for n in range(0, len(fund_codes)):
				fund_values[n] = [last_pv * ws[n]]

			position_dates.append(end_date)
			position_datas.append(ws)


		pv = 0
		d = dates[i]
		for n in range(0, len(fund_codes)):
			vs = fund_values[n]
			code = fund_codes[n]
			fund_last_v = vs[-1]
			fund_last_v = fund_last_v + fund_last_v * dfr.loc[d, code]
			vs.append(fund_last_v)
			pv = pv + vs[-1]
		portfolio_vs.append(pv)
		result_dates.append(d)

		print d , pv


	result_datas  = portfolio_vs
	result_df = pd.DataFrame(result_datas, index=result_dates,
							 columns=['low_risk_asset'])

	result_df.to_csv('./tmp/lowriskasset.csv')


	lowriskposition_df = pd.DataFrame(position_datas, index=position_dates, columns=dfr.columns)
	lowriskposition_df.to_csv('./tmp/lowriskposition.csv')


	return result_df


def highlowallocation(dfr, his_week, interval):

	result_dates = []
	result_datas = []

	position_dates = []
	position_datas = []

	dates        = dfr.index

	portfolio_vs = [1]
	result_dates.append(dates[his_week])

	fund_values  = {}
	fund_codes   = []

	for i in range(his_week + 1, len(dates)):


		if (i - his_week - 1 ) % interval == 0:

			start_date = dates[i- his_week].strftime('%Y-%m-%d')
			end_date   = dates[i - 1].strftime('%Y-%m-%d')

			allocation_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d')]
			allocation_dfr = allocation_dfr[allocation_dfr.index >= datetime.strptime(start_date, '%Y-%m-%d')]

			risk, returns, ws, sharpe = pf.markowitz_r(allocation_dfr, None)
			fund_codes = allocation_dfr.columns


			last_pv = portfolio_vs[-1]
			fund_values = {}
			for n in range(0, len(fund_codes)):
				fund_values[n] = [last_pv * ws[n]]

			position_dates.append(end_date)
			position_datas.append(ws)
		pv = 0
		d = dates[i]
		for n in range(0, len(fund_codes)):
			vs = fund_values[n]
			code = fund_codes[n]
			fund_last_v = vs[-1]
			fund_last_v = fund_last_v + fund_last_v * dfr.loc[d, code]
			vs.append(fund_last_v)
			pv = pv + vs[-1]
		portfolio_vs.append(pv)
		result_dates.append(d)

		print d , pv


	result_datas  = portfolio_vs
	result_df = pd.DataFrame(result_datas, index=result_dates,
							 columns=['highlow_risk_asset'])

	result_df.to_csv('./tmp/highlowriskasset.csv')


	highlowriskposition_df = pd.DataFrame(position_datas, index=position_dates, columns=dfr.columns)
	highlowriskposition_df.to_csv('./tmp/highlowriskposition.csv')


	return result_df


if __name__ == '__main__':


	highriskassetdf  = pd.read_csv('./tmp/equalriskasset.csv', index_col = 'date', parse_dates = 'date' )
	highriskassetdfr = highriskassetdf.pct_change().fillna(0.0)


	lowassetlabel    = ['ratebond','creditbond']
	lowriskassetdfr  = pd.read_csv('./tmp/labelasset.csv', index_col = 'date', parse_dates = 'date' )
	lowriskassetdfr  = lowriskassetdfr[lowassetlabel]
	lowriskassetdfr  = lowriskassetdfr.loc[highriskassetdfr.index]


	his_week = 13
	interval = 5
	highdf = highriskasset(highriskassetdfr, his_week, interval)
	lowdf  = lowriskasset(lowriskassetdfr, his_week, interval)


	df  = pd.concat([highdf, lowdf], axis = 1, join_axes=[highdf.index])
	dfr = df.pct_change().fillna(0.0)


	highlowdf = highlowallocation(dfr, his_week, interval)


	print "sharpe : ", fi.portfolio_sharpe(highlowdf['highlow_risk_asset'].values)
	print "annual_return : ", fi.portfolio_return(highlowdf['highlow_risk_asset'].values)
	print "maxdrawdown : ", fi.portfolio_maxdrawdown(highlowdf['highlow_risk_asset'].values)


	#print "sharpe : ", fi.portfolio_sharpe(highdf['high_risk_asset'].values)
	#print "annual_return : ", fi.portfolio_return(highdf['high_risk_asset'].values)
	#print "maxdrawdown : ", fi.portfolio_maxdrawdown(highdf['high_risk_asset'].values)


	#print lowriskassetdfr
