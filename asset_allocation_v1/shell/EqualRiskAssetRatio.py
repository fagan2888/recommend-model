#coding=utf8


import string
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import sys
sys.path.append('shell')


def riskmeanstd(risks):

	risk_mean = np.mean(risks)
	risk_std  = np.std(risks)

	rerisk = []
	risk_max = risk_mean + 2 * risk_std
	risk_min = risk_mean - 2 * risk_std

	for risk in risks:
		if risk >= risk_max or risk <= risk_min or np.isnan(risk):
			continue
		rerisk.append(risk)

	return np.mean(rerisk), np.std(rerisk)



def equalriskassetratio(allocationdata):

	assetlabels = ['largecap','smallcap','rise','oscillation','decline','growth','value','convertiblebond','SP500.SPI','GLNC','HSCI.HI']

	dfr         = allocationdata.label_asset_df
	#dfr         = pd.read_csv('./tmp/labelasset.csv', index_col = 'date', parse_dates = 'date' )
	dates = dfr.index


	interval = allocationdata.fixed_risk_asset_risk_adjust_period
	his_week = allocationdata.fixed_risk_asset_risk_lookback

	result_dates = []
	result_datas  = []


	for i in range(his_week, len(dates)):

		d = dates[i]

		#if (i - his_week) % interval == 0:
		if i % interval == 0:

			start_date = dates[i - his_week].strftime('%Y-%m-%d')
			end_date   = dates[i].strftime('%Y-%m-%d')
			allocation_start_date = dates[i - interval].strftime('%Y-%m-%d')

			allocation_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d').date()]
			allocation_dfr = allocation_dfr[allocation_dfr.index >= datetime.strptime(allocation_start_date, '%Y-%m-%d').date()]

			his_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d').date()]
			his_dfr = his_dfr[his_dfr.index >= datetime.strptime(start_date, '%Y-%m-%d').date()]

			j = 0
			risks = {}
			while j <= len(his_dfr.index):

				riskdfr = his_dfr.iloc[j:j + interval]
				#print riskdfr

				risk_data = {}
				for code in riskdfr.columns:
					risk_data[code] = np.std(riskdfr[code].values)

				for k,v in risk_data.items():
					risk = risks.setdefault(k, [])
					risk.append(v)

				j = j + interval


			ratio_data = []
			for asset in assetlabels:
				mean, std = riskmeanstd(risks[asset])
				asset_std = np.std(allocation_dfr[asset].values)

				max_risk  = mean + 2 * std
				#print mean, std, asset_std, max_risk


				position = 0
				if asset_std >= max_risk:
					position = 0.0
				elif asset_std <= mean:
					position = 1.0
				else:
					position = mean / asset_std
				ratio_data.append(position)


				#print d, asset, position


			result_datas.append(ratio_data)
			result_dates.append(d)


	result_df = pd.DataFrame(result_datas, index=result_dates, columns=assetlabels)
	result_df.index.name = 'date'

	result_df.to_csv('./tmp/equalriskassetratio.csv')

	allocationdata.equal_risk_asset_ratio_df = result_df


