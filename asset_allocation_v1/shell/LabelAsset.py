#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import Financial as FIN
import StockTag as ST
import Data
from numpy import *
import datetime
import FundFilter
import FundIndicator
import FundSelector
import Const
import pandas as pd
import time
import AllocationData


def stockLabelAsset(dates, interval, funddf, indexdf):

	df       = Data.funds()
	dfr      = df.pct_change().fillna(0.0)

	funddfr  = funddf.pct_change().fillna(0.0)
	indexdfr = indexdf.pct_change().fillna(0.0)

	tag = {}

	result_dates = []
	columns      = []
	result_datas = []
	select_datas = []

	fund_dates = []
	fund_datas = []

	allcodes    = []
	filtercodes = []
	poolcodes   = []
	selectcodes = []


	for i in range(52, len(dates)):

		if (i - 52) % interval == 0:

			start_date                    = dates[i - 52].strftime('%Y-%m-%d')
			end_date                      = dates[i].strftime('%Y-%m-%d')
			allocation_start_date         = dates[i - interval].strftime('%Y-%m-%d')

			allocationdf = Data.fund_value(allocation_start_date, end_date)
			alldf        = Data.fund_value(start_date, end_date)

			#print
			#print time.time()
			codes, indicator     = FundFilter.stockfundfilter(start_date, end_date)

			#print time.time()
			fund_pool, fund_tags = ST.tagstockfund(start_date, end_date, codes)

			#print time.time()
			#print
			allocationdf   = allocationdf[fund_pool]
			fund_code, tag = FundSelector.select_stock(allocationdf, fund_tags)


			allcodes    = alldf.columns
			filtercodes = codes
			poolcodes   = fund_pool
			selectcodes = fund_code


			#print tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']
			#print tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']


			fund_dates.append(end_date)
			fund_datas.append([tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']])


		d = dates[i]
		result_dates.append(d)
		result_datas.append([funddfr.loc[d, tag['largecap']], funddfr.loc[d, tag['smallcap']], funddfr.loc[d, tag['rise']], funddfr.loc[d, tag['oscillation']], funddfr.loc[d, tag['decline']], funddfr.loc[d, tag['growth']], funddfr.loc[d, tag['value']]] )
		print d.strftime('%Y-%m-%d'), funddfr.loc[d, tag['largecap']], funddfr.loc[d, tag['smallcap']], funddfr.loc[d, tag['rise']], funddfr.loc[d, tag['oscillation']], funddfr.loc[d, tag['decline']], funddfr.loc[d, tag['growth']], funddfr.loc[d, tag['value']]


		allcode_r = 0
		for code in allcodes:
			allcode_r = allcode_r + 1.0 / len(allcodes) * dfr.loc[d, code]


		filtercode_r = 0
		for code in filtercodes:
			filtercode_r = filtercode_r + 1.0 / len(filtercodes) * dfr.loc[d, code]


		poolcode_r = 0
		for code in poolcodes:
			poolcode_r = poolcode_r + 1.0 / len(poolcodes) * dfr.loc[d, code]

		selectcode_r = 0
		for code in selectcodes:
			selectcode_r = selectcode_r + 1.0 / len(selectcodes) * dfr.loc[d, code]

		select_datas.append([allcode_r, filtercode_r, poolcode_r, selectcode_r])


	result_df = pd.DataFrame(result_datas, index = result_dates, columns=['largecap', 'smallcap', 'rise', 'oscillation', 'decline', 'growth', 'value'])
	result_df.to_csv('./tmp/stocklabelasset.csv')

	select_df = pd.DataFrame(select_datas, index = result_dates, columns=['allcodes','filtercodes','poolcode','selectcode'])
	select_df.to_csv('./tmp/stockselectasset.csv')

	fund_df = pd.DataFrame(fund_datas , index = fund_dates, columns=['largecap', 'smallcap', 'rise', 'oscillation', 'decline', 'growth', 'value'])
	fund_df.index.name = 'date'
	fund_df.to_csv('./tmp/stock_fund.csv')


	AllocationData.stock_fund_df = fund_df

	return result_df



def bondLabelAsset(dates, interval, funddf, indexdf):

	df  = Data.bonds()
	dfr = df.pct_change().fillna(0.0)


	funddfr = funddf.pct_change().fillna(0.0)
	indexdfr = indexdf.pct_change().fillna(0.0)


	pre_ratebond        = ''
	pre_creditbond      = ''
	pre_convertiblebond = ''


	tag = {}
	result_dates = []
	columns = []
	result_datas = []


	fund_dates = []
	fund_datas = []

	select_datas = []

	allcodes    = []
	filtercodes = []
	poolcodes   = []
	selectcodes = []


	for i in range(interval + 52, len(dates)):

		if (i - 52) % interval == 0:

			start_date = dates[i - 52].strftime('%Y-%m-%d')
			end_date = dates[i].strftime('%Y-%m-%d')
			allocation_start_date = dates[i - interval].strftime('%Y-%m-%d')


			allocationdf = Data.bond_value(allocation_start_date, end_date)
			alldf        = Data.bond_value(start_date, end_date)


			codes, indicator     = FundFilter.bondfundfilter(start_date, end_date)
			fund_pool, fund_tags = ST.tagbondfund(start_date, end_date, codes)

			allocationdf   = allocationdf[fund_pool]
			fund_code, tag = FundSelector.select_bond(allocationdf, fund_tags)


			allcodes    = alldf.columns
			filtercodes = codes
			poolcodes   = fund_pool
			selectcodes = fund_code


			if not tag.has_key('ratebond'):
				tag['ratebond'] = pre_ratebond
			else:
				pre_ratebond    = tag['ratebond']
			if not tag.has_key('creditbond'):
				tag['creditbond'] = pre_creditbond
			else:
				pre_creditbond  = tag['creditbond']
			if not tag.has_key('convertiblebond'):
				tag['convertiblebond'] = pre_convertiblebond
			else:
				pre_convertiblebond = tag['convertiblebond']


			print tag['ratebond'], tag['creditbond'], tag['convertiblebond']
			# print tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']


			fund_dates.append(end_date)
			fund_datas.append([tag['ratebond'] , tag['creditbond'], tag['convertiblebond']])


		d = dates[i]
		result_dates.append(d)
		result_datas.append(
			[funddfr.loc[d, tag['ratebond']], funddfr.loc[d, tag['creditbond']], funddfr.loc[d, tag['convertiblebond']]])
		print d.strftime('%Y-%m-%d'), funddfr.loc[d, tag['ratebond']], funddfr.loc[d, tag['creditbond']], funddfr.loc[d, tag['convertiblebond']]


		allcode_r = 0
		for code in allcodes:
			allcode_r = allcode_r + 1.0 / len(allcodes) * dfr.loc[d, code]

		filtercode_r = 0
		for code in filtercodes:
			filtercode_r = filtercode_r + 1.0 / len(filtercodes) * dfr.loc[d, code]


		poolcode_r = 0
		for code in poolcodes:
			poolcode_r = poolcode_r + 1.0 / len(poolcodes) * dfr.loc[d, code]


		selectcode_r = 0
		for code in selectcodes:
			selectcode_r = selectcode_r + 1.0 / len(selectcodes) * dfr.loc[d, code]


		select_datas.append([allcode_r, filtercode_r, poolcode_r, selectcode_r])

	result_df = pd.DataFrame(result_datas, index=result_dates,
							 columns=['ratebond', 'creditbond', 'convertiblebond'])
	result_df.to_csv('./tmp/bondlabelasset.csv')

	select_df = pd.DataFrame(select_datas, index = result_dates, columns=['allcodes','filtercodes','poolcode','selectcode'])
	select_df.to_csv('./tmp/bondselectasset.csv')


	fund_df = pd.DataFrame(fund_datas , index = fund_dates, columns=['ratebond', 'creditbond','convertiblebond'])
	fund_df.index.name = 'date'
	fund_df.to_csv('./tmp/bond_fund.csv')
	
	AllocationData.bond_fund_df = fund_df

	return result_df


def moneyLabelAsset(dates, interval, funddf, indexdf):

	funddfr = funddf.pct_change().fillna(0.0)

	tag = {}
	result_dates = []
	columns = []
	result_datas = []

	for i in range(interval + 52, len(dates)):

		if (i - 52) % interval == 0:

			start_date = dates[i - 52].strftime('%Y-%m-%d')
			end_date = dates[i].strftime('%Y-%m-%d')
			allocation_start_date = dates[i - interval].strftime('%Y-%m-%d')

			allocation_funddf = Data.money_value(allocation_start_date, end_date)
			fund_codes, tag   = FundSelector.select_money(allocation_funddf)

		print tag
		# print tag
		# print fund_codes

		d = dates[i]
		result_dates.append(d)
		result_datas.append(
			[funddfr.loc[d, tag['money']]])

		print d.strftime('%Y-%m-%d'), ',', funddfr.loc[d, tag['money']]

	result_df = pd.DataFrame(result_datas, index=result_dates,columns=['money'])
	result_df.to_csv('./tmp/moneylabelasset.csv')

	return result_df


def otherLabelAsset(dates, interval, funddf, indexdf):

	funddfr = funddf.pct_change().fillna(0.0)

	result_dates = []
	columns = []
	result_datas = []

	for i in range(interval + 52, len(dates)):

		d = dates[i]

		result_dates.append(d)
		result_datas.append(
			[funddfr.loc[d, 'SP500.SPI'], funddfr.loc[d, 'SPGSGCTR.SPI'], funddfr.loc[d, 'HSCI.HI']] )

		print d.strftime('%Y-%m-%d'), ',', funddfr.loc[d, 'SP500.SPI'], ',', funddfr.loc[d, 'SPGSGCTR.SPI'], ',' , funddfr.loc[d, 'HSCI.HI']

	result_df = pd.DataFrame(result_datas, index=result_dates,
							 columns=['SP500.SPI', 'SPGSGCTR.SPI', 'HSCI.HI'])

	result_df.to_csv('./tmp/otherlabelasset.csv')
	return result_df



def labelasset(start_date, end_date):


	indexdf = Data.index_value(start_date, end_date, '000300.SH')
	dates = indexdf.pct_change().index

	allfunddf = Data.funds()

	#his_week = 156
	interval = 26

	stock_df = stockLabelAsset(dates, interval, allfunddf, indexdf)

	bondindexdf = Data.bond_index_value(start_date, end_date, Const.csibondindex_code)
	allbonddf   = Data.bonds()
	bond_df = bondLabelAsset(dates, interval, allbonddf, bondindexdf)

	allmoneydf  = Data.moneys()
	#print allmoneydf
	money_df = moneyLabelAsset(dates, interval, allmoneydf, None)

	allotherdf  = Data.others()
	other_df = otherLabelAsset(dates, interval, allotherdf, None)

	df = pd.concat([stock_df, bond_df, money_df, other_df], axis = 1, join_axes=[stock_df.index])

	df.index.name = 'date'

	df = df.dropna()

	AllocationData.label_asset_df = df

	df.to_csv('./tmp/labelasset.csv')



if __name__ == '__main__':


	start_date = '2007-01-05'
	end_date = '2016-04-22'

	indexdf = data.index_value(start_date, end_date, '000300.SH')
	dates = indexdf.pct_change().index

	allfunddf = data.funds()

	#his_week = 156
	interval = 26

	stock_df = stockLabelAsset(dates, interval, allfunddf, indexdf)

	bondindexdf = data.bond_index_value(start_date, end_date, const.csibondindex_code)
	allbonddf   = data.bonds()
	bond_df = bondLabelAsset(dates, interval, allbonddf, bondindexdf)

	allmoneydf  = data.moneys()
	#print allmoneydf
	money_df = moneyLabelAsset(dates, interval, allmoneydf, None)

	allotherdf  = data.others()
	other_df = otherLabelAsset(dates, interval, allotherdf, None)

	df = pd.concat([stock_df, bond_df, money_df, other_df], axis = 1, join_axes=[stock_df.index])

	df.to_csv('./tmp/labelasset.csv')

