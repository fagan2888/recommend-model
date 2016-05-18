#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import Financial as fin
import stocktag as st
import data
from numpy import *
import datetime
import FundFilter
import fundindicator
import fund_selector
import const
import pandas as pd


def stockLabelAsset(dates, interval, funddf, indexdf):


	df       = data.funds()
	dfr      = df.pct_change().fillna(0.0)


	funddfr  = funddf.pct_change().fillna(0.0)
	indexdfr = indexdf.pct_change().fillna(0.0)

	tag = {}

	result_dates = []
	columns      = []
	result_datas = []
	select_datas = []

	allcodes    = []
	filtercodes = []
	poolcodes   = []
	selectcodes = []


	for i in range(interval + 156, len(dates)):

		if (i - 156) % interval == 0:

			start_date                    = dates[i - 52].strftime('%Y-%m-%d')
			end_date                      = dates[i].strftime('%Y-%m-%d')
			allocation_start_date         = dates[i - interval].strftime('%Y-%m-%d')

			allocationdf = data.fund_value(allocation_start_date, end_date)
			alldf     = data.fund_value(start_date, end_date)

			codes, indicator     = FundFilter.stockfundfilter(start_date, end_date)

			fund_pool, fund_tags = st.tagstockfund(start_date, end_date, codes)

			allocationdf = allocationdf[fund_pool]
			fund_code, tag = fund_selector.select_stock(allocationdf, fund_tags)


			allcodes    = alldf.columns
			filtercodes = codes
			poolcodes   = fund_pool
			selectcodes = fund_code


			#print tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']


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


	return result_df


def bondLabelAsset(dates, interval, funddf, indexdf):


	df = data.bonds()
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


	select_datas = []

	allcodes    = []
	filtercodes = []
	poolcodes   = []
	selectcodes = []



	for i in range(interval + 156, len(dates)):

		if (i - 156) % interval == 0:

			start_date = dates[i - 52].strftime('%Y-%m-%d')
			end_date = dates[i].strftime('%Y-%m-%d')
			allocation_start_date = dates[i - interval].strftime('%Y-%m-%d')


			allocationdf = data.bond_value(allocation_start_date, end_date)
			alldf     = data.bond_value(start_date, end_date)


			codes, indicator = FundFilter.bondfundfilter(start_date, end_date)
			fund_pool, fund_tags = st.tagbondfund(start_date, end_date, codes)

			allocationdf = allocationdf[fund_pool]
			fund_code, tag = fund_selector.select_bond(allocationdf, fund_tags)


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

			print tag
			# print tag['largecap'] , tag['smallcap'], tag['rise'], tag['oscillation'], tag['decline'], tag['growth'], tag['value']

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


	return result_df


def moneyLabelAsset(dates, interval, funddf, indexdf):

	funddfr = funddf.pct_change().fillna(0.0)

	tag = {}
	result_dates = []
	columns = []
	result_datas = []

	for i in range(interval + 156, len(dates)):

		if (i - 156) % interval == 0:

			start_date = dates[i - 52].strftime('%Y-%m-%d')
			end_date = dates[i].strftime('%Y-%m-%d')
			allocation_start_date = dates[i - interval].strftime('%Y-%m-%d')

			allocation_funddf = data.money_value(allocation_start_date, end_date)
			fund_codes, tag = fund_selector.select_money(allocation_funddf)

		print tag
		# print tag
		# print fund_codes


		d = dates[i]
		result_dates.append(d)
		result_datas.append(
			[funddfr.loc[d, tag['sharpe1']], funddfr.loc[d, tag['sharpe2']] ])

		print d.strftime('%Y-%m-%d'), ',', funddfr.loc[d, tag['sharpe1']], ',', funddfr.loc[d, tag['sharpe2']]

	result_df = pd.DataFrame(result_datas, index=result_dates,
							 columns=['sharpe1', 'sharpe2'])

	result_df.to_csv('./tmp/moneylabelasset.csv')

	return result_df


def otherLabelAsset(dates, interval, funddf, indexdf):

	funddfr = funddf.pct_change().fillna(0.0)

	result_dates = []
	columns = []
	result_datas = []

	for i in range(interval + 156, len(dates)):

		d = dates[i]

		result_dates.append(d)
		result_datas.append(
			[funddfr.loc[d, 'SP500.SPI'], funddfr.loc[d, 'SPGSGCTR.SPI'], funddfr.loc[d, 'HSCI.HI']] )

		print d.strftime('%Y-%m-%d'), ',', funddfr.loc[d, 'SP500.SPI'], ',', funddfr.loc[d, 'SPGSGCTR.SPI'], ',' , funddfr.loc[d, 'HSCI.HI']

	result_df = pd.DataFrame(result_datas, index=result_dates,
							 columns=['SP500.SPI', 'SPGSGCTR.SPI', 'HSCI.HI'])


	result_df.to_csv('./tmp/otherlabelasset.csv')

	return result_df


if __name__ == '__main__':


	start_date = '2007-01-05'
	end_date = '2016-04-22'



	indexdf = data.index_value(start_date, end_date, '000300.SH')
	dates = indexdf.pct_change().index

	allfunddf = data.funds()

	his_week = 156
	interval = 13

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
