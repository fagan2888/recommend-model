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



def indicator():

	#code = '000001.SH'
	#df = pd.read_csv('./data/000001.csv', index_col = 'date', parse_dates = ['date'])

	his_back = 1500

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
	#code = '000001.SH'
	df = pd.read_csv('./data/000905.csv', index_col = 'date', parse_dates = ['date'])
	#df = pd.read_csv('./data/000001.csv', index_col = 'date', parse_dates = ['date'])
	ma5_df  = pd.rolling_mean(df[code], 5)
	ma10_df = pd.rolling_mean(df[code], 10)
	ma30_df = pd.rolling_mean(df[code], 30)
	ma60_df = pd.rolling_mean(df[code], 60)


	df['ma5']         = ma5_df
	df['ma10']        = ma10_df
	df['ma30']        = ma30_df
	df['ma60']        = ma60_df


	df = df.dropna()
	dfr = df.pct_change().fillna(0.0)
	#dfr.to_csv('ma.csv')

	dates = dfr.index

	risk10_datas = []
	risk25_datas = []
	risk30_datas = []
	rmean10_datas= []
	rmean25_datas= []
	rmean30_datas= []
	risk_dates   = []
	rmean10_threshold_datas = []	
	

	for i in range(30, len(dfr.index)):	

		tmp_r = dfr.iloc[i - 10 : i , 0].values
		risk10_datas.append(np.std(tmp_r))
		rmean10_datas.append(np.mean(tmp_r))


		rmean10_datas_copy = copy.copy(rmean10_datas)
		rmean10_datas_copy.sort()	
		rmean10_threshold_datas.append(rmean10_datas_copy[ (int)(0.1 * len(rmean10_datas_copy)) ] )
		print rmean10_threshold_datas[-1]
	
		tmp_r = dfr.iloc[i - 25 : i , 0].values
		risk25_datas.append(np.std(tmp_r))
		rmean25_datas.append(np.mean(tmp_r))

		tmp_r = dfr.iloc[i - 30 : i , 0].values
		risk30_datas.append(np.std(tmp_r))
		rmean30_datas.append(np.mean(tmp_r))
	
		risk_dates.append(dfr.index[i])


	riskmean_df   = pd.DataFrame(np.matrix([rmean10_datas, rmean25_datas, rmean30_datas, risk10_datas, risk25_datas, risk30_datas]).T, index = risk_dates, columns = ['rmean10','rmean25', 'rmean30', 'risk10', 'risk25','risk30'])
	df = pd.concat([df, riskmean_df], axis = 1, join_axes = [riskmean_df.index])


	print df



	'''

	start_index   = 1500

	risk_lookback = 30
	r_lookback    = 10
	
	his_back      = 300
	
	his_risks     = []
	positions      = [1]


	risks         = []
	dates = dfr.index
	position_dates = [dates[start_index]]


	for i in range(start_index, len(dates)):

		d = dates[i]
		#print dates[i]

		#r = dfr.loc[d, 'ma10']
		#rs = dfr.iloc[0:i, 1].values
		#rs.sort()
		#r_threshold = rs[(int)(0.1*len(rs))]


		now_risk_mean = np.std(dfr.iloc[i - risk_lookback:i + 1, 0])

		#all_risk_mean = np.mean(dfr.iloc[0 : i + 1, 0])


		all_risk_std  = np.std(dfr.iloc[0 : i + 1, 0])
		risks.append(all_risk_std)


		ma10 = dfr.loc[d, 'ma10']
		ma30 = dfr.loc[d, 'ma30']
		ma60 = dfr.loc[d, 'ma60']


		#print all_risk_std
		if now_risk_mean >= all_risk_std + 1 * np.std(risks) and ma10 < 0.0:	
		#if ma10 < 0.0 and ma30 < 0.0:	
			if not (0.0 == positions[-1]):
				#print d, now_risk_mean, all_risk_std, all_risk_std + np.std(risks), ma10, ma30, 0
				positions.append(0.0)
				position_dates.append(d)
		#elif ma10 > 0.0 and ma30 > 0.0  and now_risk_mean <= all_risk_std:
		elif ma10 > 0.0 and ma30 > 0.0:
			if not (1.0 == positions[-1]):
				#print d, now_risk_mean, all_risk_std, all_risk_std + np.std(risks), ma10, ma30, 1
				positions.append(1.0)
				position_dates.append(d)
		#elif now_risk_mean <= all_risk_std and ma10 > 0.0 and ma30 > 0.0:
		#	if not (1.0 == positions[-1]):
				#print d, now_risk_mean, all_risk_std, all_risk_std + np.std(risks), ma10, ma30, 1
		#		positions.append(1.0)
		#		position_dates.append(d)

	p_df = pd.DataFrame(positions, index = position_dates, columns = [code])
	print p_df

	#df = pd.read_csv('./data/000001.csv', index_col = 'date', parse_dates = ['date'])
	df = pd.read_csv('./data/000905.csv', index_col = 'date', parse_dates = ['date'])
        dfr = df.pct_change().fillna(0.0)
        equal_asset_df     = EqualRiskAsset.equalriskasset(p_df, dfr)


        asset_df = equal_asset_df[[code]]
        print asset_df
        print "sharpe : ", FundIndicator.portfolio_sharpe_day(asset_df[code].values)
        print "annual_return : ", FundIndicator.portfolio_return_day(asset_df[code].values)
        print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(asset_df[code].values)


	p_df.to_csv('position.csv')
	asset_df.to_csv('asset.csv')

	'''
