#coding=utf8


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


<<<<<<< HEAD

def risk_asset_allocation(pname='', phnum='', plnum='', pdebug='y'):
	ori_file_name = 'gaopeng.csv'
	ori_file_dir = './data/'
	out_file_dir = './result/'
	out_err_name = 'equal_risk_ratio.csv'
	out_era_name = 'equal_risk_asset.csv'
	out_ap_name = 'asset_position.csv'
	out_as_name = 'asset.csv'
	out_rp_name = 'risk_position.csv'
	out_po_name = 'position.csv'
	stock_num = 3
	bond_num  = 2

	if pdebug == 'n':
		stock_num = string.atoi(phnum)
		bond_num = string.atoi(plnum)
		ori_file_name = pname + '.csv'
		ori_file_dir = '/tmp/assets/'
		out_file_dir = '/tmp/assets/'
		out_err_name = pname + 'equal_risk_ratio.csv'
		out_era_name = pname + 'equal_risk_asset.csv'
		out_ap_name = pname + 'asset_position.csv'
		out_as_name = pname + 'asset.csv'
		out_rp_name = pname + 'risk_position.csv'
		out_po_name = pname + 'position.csv'

        df = pd.read_csv(ori_file_dir + ori_file_name, index_col = 'date', parse_dates = ['date'])
=======
def risk_asset_allocation():

	stock_num = 3
	bond_num  = 2

        df = pd.read_csv('./data/kunge.csv', index_col = 'date', parse_dates = 'date')
	df = df[df.columns[0:stock_num]]
>>>>>>> fix bug

        dfr = df.pct_change().fillna(0.0)

        week_df  = df.resample('W-FRI').last()
        week_dfr = week_df.pct_change().fillna(0.0)

        #print week_df
	
        #equal_ratio_df     = EqualRiskAssetRatio.equalriskassetratio(week_dfr)

<<<<<<< HEAD
        equal_ratio_df     = EqualRiskAssetRatio.equalriskassetratio(week_dfr, pname, pdebug)
        equal_asset_df     = EqualRiskAsset.equalriskasset(equal_ratio_df, dfr, pname, pdebug)
=======
        equal_ratio_df     = EqualRiskAssetRatio.equalriskassetratio(dfr)
        equal_asset_df     = EqualRiskAsset.equalriskasset(equal_ratio_df, dfr)
>>>>>>> fix bug

	equal_ratio_df.to_csv(out_file_dir + out_err_name)
	equal_asset_df.to_csv(out_file_dir + out_era_name)

	#asset_df = equal_asset_df[['000905.SH']]	

	#print asset_df

	#print "sharpe : ", FundIndicator.portfolio_sharpe_day(asset_df['000905.SH'].values)
	#print "annual_return : ", FundIndicator.portfolio_return_day(asset_df['000905.SH'].values)
	#print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(asset_df['000905.SH'].values)

	df = pd.concat([equal_asset_df[equal_asset_df.columns[0:stock_num]], df[df.columns[stock_num: stock_num + bond_num]] ], axis = 1, join_axes=[equal_asset_df.index])

	#df = df[df.columns[0:stock_num]]
	#df = df[ df.columns[stock_num:stock_num + bond_num] ]
	df = df[ df.columns[0 : stock_num] ]
	dfr = df.pct_change().fillna(0.0)
	#print df

	week_df  = df.resample('W-FRI').last()
	week_dfr = week_df.pct_change().fillna(0.0)

	#print equal_asset_df
	asset_position_df          = Allocation.allocation_ratio(week_dfr)
	asset_df                   = Allocation.allocation_asset(asset_position_df, dfr)

	#asset_df = equal_asset_df[ equal_asset_df.columns[0 : stock_num] ]

	asset_df, risk_position_df  = RiskControl.risk_control(asset_df)

	asset_position_df.to_csv(out_file_dir + out_ap_name)
	asset_df.to_csv(out_file_dir + out_as_name)
	risk_position_df.to_csv(out_file_dir + out_rp_name)

<<<<<<< HEAD
	print asset_df
=======
	print asset_df	

>>>>>>> fix bug
	print "sharpe : ", FundIndicator.portfolio_sharpe_day(asset_df['nav'].values)
	print "annual_return : ", FundIndicator.portfolio_return_day(asset_df['nav'].values)
	print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(asset_df['nav'].values)

	#position_df = Position.position(equal_ratio_df, asset_position_df, risk_position_df)
	#print position_df
<<<<<<< HEAD
	position_df.to_csv(out_file_dir + out_po_name)
=======
	#position_df.to_csv('./result/position.csv')
>>>>>>> fix bug
	#print asset_position_df

	return 0


if __name__ == '__main__':
	debug = 'y'
	fname = ''
	hnum = ''
	lnum = ''
	if len(sys.argv) >= 4:
		fname = sys.argv[1]
		hnum = sys.argv[2]
		lnum = sys.argv[3]
		debug = sys.argv[4]
	if debug == 'n':
		risk_asset_allocation(pname=fname, phnum=hnum, plnum=lnum, pdebug=debug)
	else:
		risk_asset_allocation()
