#coding=utf8


import pandas as pd




def risk_position(fund_df, equalrisk_ratio_df, allocation_ratio_df):

	return 0



if __name__ == '__main__':

	fund_df = pd.read_csv('./tmp/stock_fund.csv', index_col = 'date', parse_dates = ['date'])
	equalrisk_ratio_df = pd.read_csv('./tmp/equalriskassetratio.csv', index_col = 'date', parse_dates = ['date'])
	highriskposition_ratio_df = pd.read_csv('./tmp/highriskposition.csv', index_col = 'date', parse_dates = ['date'])
	#print equalrisk_ratio_df
	#print highriskposition_ratio_df
	start_date = highriskposition_ratio_df.index[0]
	print start_date
	equalrisk_ratio_df = equalrisk_ratio_df[equalrisk_ratio_df.index >= start_date]
	print equalrisk_ratio_df

	#print fund_df

	
