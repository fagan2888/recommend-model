#coding=utf8


import string
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import sys
import os
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



def equalriskassetratio(dfr, pname='', debug='y'):

	#assetlabels = ['largecap','smallcap','rise','oscillation','decline','growth','value','convertiblebond','SP500.SPI','GLNC','HSCI.HI']
	assetlabels = dfr.columns
	#dfr         = df.pct_change().fillna(0.0)

	#dfr         = pd.read_csv('./tmp/labelasset.csv', index_col = 'date', parse_dates = 'date' )
	dates = dfr.index

	interval = 5
	his_week = 30 #kunge
	#his_week = 13   #gaopeng


	result_dates = []
	result_datas  = []
	#print dates
	#print dates[0]
	#os._exit(0)
	# len(dates) = 89
	for i in range(his_week, len(dates)):### his_week to hisweek-1

		d = dates[i]

		if (i - his_week) % (interval) == 0:### i - his_week to i - his_week + 1

			start_date = dates[i - his_week].strftime('%Y-%m-%d')### i - his_week to i - his_week + 1
			end_date   = dates[i].strftime('%Y-%m-%d') ### i to i - 1
			allocation_date = dates[i - interval].strftime("%Y-%m-%d")### i - interval to i - interval + 1

			allocation_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d')]
			allocation_dfr = allocation_dfr[allocation_dfr.index >= datetime.strptime(allocation_date, '%Y-%m-%d')]

			#print dfr.index
			his_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d')]
			#his_dfr = dfr[dfr.index <= datetime.strptime(allocation_date, '%Y-%m-%d')]
			his_dfr = his_dfr[his_dfr.index >= datetime.strptime(start_date, '%Y-%m-%d')]
			print start_date,end_date
			print his_dfr,len(his_dfr)

			j = 0
			risks = {}
			#每只基金每5周的标准差
			while j <= len(his_dfr.index):

				riskdfr = his_dfr.iloc[j:j + interval]
				print riskdfr

				risk_data = {}
				for code in riskdfr.columns:
					risk_data[code] = np.std(riskdfr[code].values)

				for k,v in risk_data.items():
					risk = risks.setdefault(k, [])
					risk.append(v)

				j = j + interval

			print risk_data
			print risks
			os._exit(0)
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
	if debug == 'y':
		result_df.to_csv('./result/equalriskassetratio.csv')
	#else:
	#	result_df.to_csv('/tmp/' + pname + 'equalriskassetratio.csv')

	return result_df
