#coding=utf8



import string
import json
import sys
sys.path.append('shell')
import pandas as pd
import LabelAsset
import EqualRiskAssetRatio
import EqualRiskAsset
import HighLowRiskAsset
import os
import DB
import MySQLdb
from datetime import datetime
import AllocationData
import time
import RiskHighLowRiskAsset
import FixRisk
import WeekFund2DayNav



def risk_asset_allocation():


	allocationdata = AllocationData.allocationdata()
	allocationdata.start_date                            = '2010-01-01'
	allocationdata.all_dates()

	#allocationdata.all_dates()

	#allocationdata.start_date                            = args.get('start_date')
	#allocationdata.start_date                            = '2010-01-01'
	#allocationdata.start_date                            = '2016-01-01'

	allocationdata.fund_measure_lookback                 = 52
	allocationdata.fund_measure_adjust_period            = 26
	allocationdata.jensen_ratio                          = 0.5
	allocationdata.sortino_ratio                         = 0.5	
	allocationdata.ppw_ratio                             = 0.5		
	allocationdata.stability_ratio                       = 0.5	
	allocationdata.fixed_risk_asset_lookback             = 52	
	allocationdata.fixed_risk_asset_risk_adjust_period   = 5	
	allocationdata.allocation_lookback                   = 26 
	allocationdata.allocation_adjust_period              = 13



	LabelAsset.labelasset(allocationdata)
	WeekFund2DayNav.week2day(allocationdata)
	FixRisk.fixrisk(allocationdata)

	#EqualRiskAssetRatio.equalriskassetratio(allocationdata)
	EqualRiskAsset.equalriskasset(allocationdata)
	RiskHighLowRiskAsset.highlowriskasset(allocationdata)
	#HighLowRiskAsset.highlowriskasset(allocationdata)


	#DB.fund_measure(allocationdata)
	#DB.label_asset(allocationdata)
	#DB.asset_allocation(allocationdata)
	#DB.riskhighlowriskasset(allocationdata)




if __name__ == '__main__':

	risk_asset_allocation()
