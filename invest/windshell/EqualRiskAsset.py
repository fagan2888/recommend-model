#coding=utf8


import string
import pandas as pd


ratio_df         = pd.read_csv('./tmp/equalriskassetratio.csv', index_col = 'date', parse_dates = 'date' )
dfr              = pd.read_csv('./tmp/labelasset.csv', index_col = 'date', parse_dates = 'date' )


dates = dfr.index
ratio_dates = ratio_df.index


assetlabels  = ['largecap','smallcap','rise','oscillation','decline','growth','value','convertiblebond','SP500.SPI','SPGSGCTR.SPI','HSCI.HI']
asset_values = {}
asset_ratio  = {}


for asset in assetlabels:
	asset_values.setdefault(asset, [1.0])
	asset_ratio.setdefault(asset, 0)


result_dates = []
result_datas  = []


for i in range(0, len(dates)):

	d = dates[i]
	for asset in assetlabels:
		vs = asset_values[asset]
		last_v = vs[-1]
		current_v = last_v + last_v * dfr.loc[d, asset] * asset_ratio[asset]
		vs.append(current_v)


	if d in ratio_dates:
		for asset in assetlabels:
			asset_ratio[asset] = ratio_df.loc[d, asset]


	asset_vs = [ asset_values['largecap'][-1],asset_values['smallcap'][-1],asset_values['rise'][-1],asset_values['oscillation'][-1],asset_values['decline'][-1],asset_values['growth'][-1],\
		asset_values['value'][-1],asset_values['convertiblebond'][-1],asset_values['SP500.SPI'][-1],\
		asset_values['SPGSGCTR.SPI'][-1], asset_values['HSCI.HI'][-1] ]
	result_datas.append(asset_vs)
	result_dates.append(d)


	print d, asset_values['largecap'][-1],asset_values['smallcap'][-1],asset_values['rise'][-1],asset_values['oscillation'][-1],asset_values['decline'][-1],asset_values['growth'][-1],\
		asset_values['value'][-1],asset_values['convertiblebond'][-1],asset_values['SP500.SPI'][-1],\
		asset_values['SPGSGCTR.SPI'][-1], asset_values['HSCI.HI'][-1]



	if d in ratio_dates:
		for asset in assetlabels:
			asset_ratio[asset] = ratio_df.loc[d, asset]


result_df = pd.DataFrame(result_datas, index=result_dates,
									 columns=assetlabels)

result_df.to_csv('./tmp/equalriskasset.csv')
