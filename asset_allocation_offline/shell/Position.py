#coding=utf8


import pandas as pd



def position(equal_risk_df, asset_position_df, risk_position_df):

	#print equal_risk_df.index
	#print asset_position_df.index
	#print risk_position_df.index
	#print risk_position_df.columns

	position_datas = []
	position_dates = []

	asset_position = {}
	equal_risk     = {}
	codes = asset_position_df.columns

	for code in codes:
		asset_position[code] = 0
		equal_risk[code]     = 0	
	
	dates = risk_position_df.index

	equal_risk_index     = 0
	asset_position_index = 0


	for d in dates:

		risk_position = risk_position_df.loc[d, 'position']

		if equal_risk_index < len(equal_risk_df.index) and d >= equal_risk_df.index[equal_risk_index]:
			for code in codes:
				equal_risk[code] = equal_risk_df.loc[equal_risk_df.index[equal_risk_index], code]	
			equal_risk_index = equal_risk_index + 1

		if asset_position_index < len(asset_position_df.index) and d >= asset_position_df.index[asset_position_index]:
			for code in codes:
				asset_position[code] = asset_position_df.loc[asset_position_df.index[asset_position_index], code]	
			asset_position_index = asset_position_index + 1

		ps = []	
		for code in codes:
			#print code , risk_position * equal_risk[code] * asset_position[code]			
			p = risk_position * equal_risk[code] * asset_position[code]			
			ps.append(p)

		position_datas.append(ps)
		position_dates.append(d)

	df = pd.DataFrame(position_datas, index = position_dates, columns = codes)
	df.index.name = 'date'

	return df
