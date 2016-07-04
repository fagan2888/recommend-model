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
sys.path.append('shell')


def indicator():

	code = '000905.SH'
	#code = '000001.SH'
	#df = pd.read_csv('./data/000001.csv', index_col = 'date', parse_dates = ['date'])
	his_back = 1500
	df = pd.read_csv('./data/000905.csv', index_col = 'date', parse_dates = ['date'])
	dfr = df.pct_change().fillna(0.0)
	dates = dfr.index


	risk_lookback = 30
	r_lookback    = 10


	risks      = []
	rs         = []
	ds         = []

	result_datas = []

	for i in range(his_back, len(dates)):
		d = dates[i]

		record = []
		tmp_r = [] 
		for m in range(0, risk_lookback):
			tmp_r.append(dfr.loc[dates[i-m] , code])
		risks.append(np.std(tmp_r))
		print d, np.std(tmp_r), np.mean(risks), np.std(risks),
		record.append(np.std(tmp_r))
		record.append(np.mean(risks))
		record.append(np.std(risks))

		tmp_r = [] 
		for n in range(0, r_lookback):
			tmp_r.append(dfr.loc[dates[i-n] , code])
		rs.append(np.mean(tmp_r))
		rs.sort()	
		print np.mean(tmp_r), rs[int(0.1 * len(rs))], rs[int(0.5 * len(rs))]
		record.append(np.mean(tmp_r))
		record.append(rs[int(0.1 * len(rs))])
		record.append(rs[int(0.5 * len(rs))])

		result_datas.append(record)
		#print d, 
	
 
	risk_df     = pd.DataFrame(risks, index = dates[his_back:], columns = ['risk']) 
	rs_df       = pd.DataFrame(rs,    index = dates[his_back:], columns = ['r'])
	result_df   = pd.DataFrame(result_datas,    index = dates[his_back:], columns = ['risk', 'risk_mean', 'risk_std', 'r', 'r_down_threshold','r_up_threshold'])
	
	risk_df.to_csv('./tmp/risks.csv')
	rs_df.to_csv('./tmp/rs.csv')
	result_df.to_csv('./tmp/result.csv')
	#print dates[3000]


#def risk_control():
		
	

if __name__ == '__main__':


	code = '000905.SH'
	indicator()

	df = pd.read_csv('./tmp/result.csv', index_col = 'date', parse_dates = ['date'])
	dates = df.index


	result_datas = []
	result_dates = []

	rolling = 30

	for i in range(rolling, len(dates)):
		
		d = dates[i]

		risk = df.loc[d, 'risk']
		risk_mean = df.loc[d, 'risk_mean']
		risk_std  = df.loc[d, 'risk_std']
		r         = df.loc[d, 'r']
		r_down_threshold = df.loc[d, 'r_down_threshold']
		r_up_threshold   = df.loc[d, 'r_up_threshold']

		rolling_r = 0
		for n in range(0, rolling):
			rolling_r = rolling_r + df.loc[dates[i-n], 'r']
		rolling_r = rolling_r / rolling

		position = 1
		change_position = False

		if   risk >= risk_mean + 3 * risk_std:
			position = 0
			change_position = True
		elif r <= r_down_threshold:
			if risk >= risk_mean:	
				position = 1.0 - (risk - risk_mean) / risk_std		
				if position < 0:
					position = 0
			else: 
				position = 1
			change_position = True
		elif rolling_r >= r_up_threshold:
			position = 1
			change_position = True
		elif risk <= risk_mean and r >= r_up_threshold:
			position = 1
			change_position = True

		if change_position:
			#print d, position	
			if len(result_datas) == 0:
				result_dates.append(d)
				result_datas.append(position)
			else:
				last_position = result_datas[-1]
				if last_position == 0 and position == 0:
					continue
				else:
					diff = last_position * 0.4
					if position <= last_position - diff or position >= last_position + diff:
						result_dates.append(d)
						result_datas.append(position)
					

	position_df = pd.DataFrame(result_datas, index=result_dates, columns = [code])
	position_df.to_csv('position.csv')
	position_df.index.name = 'date'

	#df = pd.read_csv('./data/000001.csv', index_col = 'date', parse_dates = ['date'])
	df = pd.read_csv('./data/000905.csv', index_col = 'date', parse_dates = ['date'])
	dfr = df.pct_change().fillna(0.0)
	equal_asset_df     = EqualRiskAsset.equalriskasset(position_df, dfr)

	asset_df = equal_asset_df[[code]]
        print asset_df
        print "sharpe : ", FundIndicator.portfolio_sharpe_day(asset_df[code].values)
        print "annual_return : ", FundIndicator.portfolio_return_day(asset_df[code].values)
        print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(asset_df[code].values)


	asset_df.to_csv('asset.csv')
