#coding=utf8


import sys
sys.path.append('shell')
import pandas as pd
import LabelAsset
import EqualRiskAssetRatio
import EqualRiskAsset
import HighLowRiskAsset


if __name__ == '__main__':

	start_date = '2006-01-05'
	end_date = '2016-04-22'

	LabelAsset.labelasset(start_date, end_date)
	EqualRiskAssetRatio.equalriskassetratio()
	EqualRiskAsset.equalriskasset()
	HighLowRiskAsset.highlowriskasset()


