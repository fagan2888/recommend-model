#coding=utf8


import string
import pandas as pd
import time
from datetime import datetime
import sys
sys.path.append('shell')


def equalriskasset(df):


	rf = 0.03 / 52


	#ratio_df         = allocationdata.equal_risk_asset_ratio_df
	ratio_df         = pd.read_csv('./tmp/equalriskassetratio.csv', index_col = 'date', parse_dates = 'date' )
	ratio_dates      = ratio_df.index
	start_date = ratio_dates[0]


	#dfr              = allocationdata.label_asset_df
	#dfr              = pd.read_csv('./tmp/labelasset.csv', index_col = 'date', parse_dates = 'date' )
	dfr               = df.pct_change().fillna(0.0)
	dfr               = dfr[dfr.index >= start_date]

	dates = dfr.index
	#assetlabels  = ['largecap','smallcap','rise','oscillation','decline','growth','value','convertiblebond','SP500.SPI','GLNC','HSCI.HI']

	assetlabels   = dfr.columns	
	#assetlabels  = ['largecap','smallcap','rise','oscillation','decline','growth','value','convertiblebond','SP500.SPI','GLNC','HSCI.HI']
	asset_values = {}
	asset_combination = {}
	asset_ratio  = {}


	for asset in assetlabels:

		asset_values.setdefault(asset, [1.0])
		asset_ratio.setdefault(asset, 0)
		asset_combination.setdefault(asset, [[0.0, 0,0]])


	result_dates = []
	result_datas  = []

	for i in range(0, len(dates)):


		d = dates[i]
		for asset in assetlabels:

			cvs = asset_combination[asset]
			last_cvs = cvs[-1]
			current_cvs = [last_cvs[0] * (1 + dfr.loc[d, asset]), last_cvs[1] * (1 + rf)]
			#current_cvs = [last_cvs[0] * (1 + dfr.loc[d, asset]), last_cvs[1]]
			cvs.append(current_cvs)
			vs = asset_values[asset]
			if current_cvs[0] + current_cvs[1] == 0:
				continue
			else:
				vs.append(current_cvs[0] + current_cvs[1])


			'''
			vs = asset_values[asset]
			last_v = vs[-1]
			current_v = last_v + last_v * dfr.loc[d, asset] * asset_ratio[asset]
			vs.append(current_v)
			'''


		if d in ratio_dates:
			for asset in assetlabels:
				asset_ratio[asset] = ratio_df.loc[d, asset]
				vs = asset_values[asset]
				cvs = asset_combination[asset]
				cvs.append( [vs[-1] * asset_ratio[asset], vs[-1] * (1 - asset_ratio[asset])] )


		#asset_vs = [asset_values['largecap'][-1], asset_values['smallcap'][-1], asset_values['rise'][-1],
		#			asset_values['oscillation'][-1], asset_values['decline'][-1], asset_values['growth'][-1], \
		#			asset_values['value'][-1], asset_values['convertiblebond'][-1], asset_values['SP500.SPI'][-1], \
		#			asset_values['GLNC'][-1], asset_values['HSCI.HI'][-1]]
		#result_datas.append(asset_vs)
		#result_dates.append(d)

		asset_vs = []
		for label in assetlabels:
			asset_vs.append(asset_values[label][-1])


		#asset_vs = [asset_values['largecap'][-1], asset_values['smallcap'][-1], asset_values['rise'][-1],
		#			asset_values['oscillation'][-1], asset_values['decline'][-1], asset_values['growth'][-1], \
		#			asset_values['value'][-1], asset_values['SP500.SPI'][-1], \
		#			asset_values['GLNC'][-1], asset_values['HSCI.HI'][-1]]
		result_datas.append(asset_vs)
		result_dates.append(d)


		print d,
		for label in assetlabels:
			print asset_values[label][-1],
		print 


		'''
		print d, asset_values['largecap'][-1], asset_values['smallcap'][-1], asset_values['rise'][-1], \
		asset_values['oscillation'][-1], asset_values['decline'][-1], asset_values['growth'][-1], \
			asset_values['value'][-1], asset_values['convertiblebond'][-1], asset_values['SP500.SPI'][-1], \
			asset_values['GLNC'][-1], asset_values['HSCI.HI'][-1]
		'''


		if d in ratio_dates:
			for asset in assetlabels:
				asset_ratio[asset] = ratio_df.loc[d, asset]



	#new_assetlabels  = ['largecap','smallcap','rise','oscillation','decline','growth','value','SP500.SPI','GLNC','HSCI.HI']
	result_df = pd.DataFrame(result_datas, index=result_dates, columns=assetlabels)

	result_df.index.name = 'date'
	result_df.to_csv('./tmp/equalriskasset.csv')
