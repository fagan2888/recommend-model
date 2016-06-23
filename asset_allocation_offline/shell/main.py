#coding=utf8


import sys
sys.path.append('shell')
import pandas as pd
import LabelAsset
import EqualRiskAssetRatio
import EqualRiskAsset
import HighLowRiskAsset
import os
import DB


if __name__ == '__main__':


	#start_date = '2010-01-05'
	#end_date = '2016-04-22'
	#LabelAsset.labelasset(start_date, end_date)


	df = pd.read_csv('./data/gaopeng.csv', index_col = 'date', parse_dates = 'date')
	#df = pd.read_csv('./data/kunge.csv', index_col = 'date', parse_dates = 'date')
	EqualRiskAssetRatio.equalriskassetratio(df)
	EqualRiskAsset.equalriskasset(df)
	HighLowRiskAsset.highlowriskasset()
