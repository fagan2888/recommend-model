#codeing=utf8


import string
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import sys
import EqualRiskAssetRatio
import EqualRiskAsset
import Allocation
import RiskControl
import Position
import copy
sys.path.append('shell')



if __name__ == '__main__':


	code = '000905.SH'
	#code = '000001.SH'
	df = pd.read_csv('./data/000905.csv', index_col = 'date', parse_dates = ['date'])
	#df = pd.read_csv('./data/000001.csv', index_col = 'date', parse_dates = ['date'])
	ma5_df  = pd.rolling_mean(df[code], 5)
	ma10_df = pd.rolling_mean(df[code], 10)
	ma20_df = pd.rolling_mean(df[code], 20)
	ma30_df = pd.rolling_mean(df[code], 30)
	ma60_df = pd.rolling_mean(df[code], 60)


	df['ma5']         = ma5_df
	df['ma10']        = ma10_df
	df['ma20']        = ma20_df
	df['ma30']        = ma30_df
	df['ma60']        = ma60_df


	df = df.dropna()
	dfr = df.pct_change().fillna(0.0)
	#dfr.to_csv('ma.csv')

	dates = dfr.index

	position_datas = []
	position_dates = []	

	for i in range(1, len(dates)):

		d = dates[i]
		r = dfr.loc[d, 'ma20']	
		if r >= 0.0:
			p = 1.0
		else:
			p = 0.0

		if len(position_datas) == 0 and p == 1.0:
			position_datas.append(p)
			position_dates.append(d)
		elif len(position_datas) > 0 and (not position_datas[-1] == p):
			position_datas.append(p)
			position_dates.append(d)

	position_df = pd.DataFrame(position_datas, index = position_dates, columns = ['000905.SH'])

	#print position_df
	#print dfr

	df = Allocation.allocation_asset(position_df, dfr[['000905.SH']])	
	#print df
	df.to_csv('./tmp/twoeight.csv')
	position_df.to_csv('./tmp/position.csv')
