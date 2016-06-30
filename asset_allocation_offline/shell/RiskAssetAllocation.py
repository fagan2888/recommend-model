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



def risk_asset_allocation():

	stock_num = 3
	bond_num  = 2

        df = pd.read_csv('./data/gaopeng.csv', index_col = 'date', parse_dates = 'date')

        dfr = df.pct_change().fillna(0.0)

        week_df  = df.resample('W-FRI').last()
        week_dfr = week_df.pct_change().fillna(0.0)

        #print week_df

        equal_ratio_df     = EqualRiskAssetRatio.equalriskassetratio(week_dfr)
        equal_asset_df     = EqualRiskAsset.equalriskasset(equal_ratio_df, dfr)

	equal_ratio_df.to_csv('./result/equal_risk_ratio.csv')
	equal_asset_df.to_csv('./result/equal_risk_asset.csv')


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


	#asset_df = df[ df.columns[1 : 2] ]
	asset_df,risk_position_df  = RiskControl.risk_control(asset_df)

	asset_position_df.to_csv('./result/asset_position.csv')
	asset_df.to_csv('./result/asset.csv')
	risk_position_df.to_csv('./result/risk_position.csv')

	print asset_df	
	print "sharpe : ", FundIndicator.portfolio_sharpe_day(asset_df['nav'].values)
	print "annual_return : ", FundIndicator.portfolio_return_day(asset_df['nav'].values)
	print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(asset_df['nav'].values)

	position_df = Position.position(equal_ratio_df, asset_position_df, risk_position_df)
	#print position_df
	position_df.to_csv('./result/position.csv')
	#print asset_position_df

	return 0


if __name__ == '__main__':

	risk_asset_allocation()
