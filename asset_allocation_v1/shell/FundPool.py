#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import Financial as fin
import Const
import Data
from numpy import isnan
from datetime import datetime
import pandas as pd
import AllocationData
import DBData
import FundFilter
import StockTag as ST
import FundIndicator
import DFUtil
import MySQLdb



def stock_fund_measure(allocationdata, start_date, end_date):


	conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='asset_allocation', charset='utf8')
        cursor = conn.cursor()

	lookback  = 52
	
	index_df  = DBData.index_value(start_date, end_date)
	stock_df  = DBData.stock_fund_value(start_date, end_date)


	codes, indicator     = FundFilter.stockfundfilter(allocationdata, stock_df, index_df[Const.hs300_code])
	fund_pool, fund_tags = ST.tagstockfund(allocationdata, stock_df[codes], index_df)


	fund_sharpe          = FundIndicator.fund_sharp_annual(stock_df[fund_pool])
	fund_jensen          = FundIndicator.fund_jensen(stock_df[fund_pool], index_df[Const.hs300_code])


	base_sql = "replace into fund_pool (fm_date, fm_look_back, fm_fund_type, fm_fund_code, fm_jensen, fm_ppw, fm_stability, fm_sortino, fm_sharpe, fm_largecap, fm_smallcap, fm_growth, fm_value, fm_rise, fm_oscillation, fm_decline, created_at, updated_at) values ('%s',%d, %d, '%s',%f, %f, %f, %f, %f, %d, %d, %d, %d, %d, %d, %d, '%s', '%s')"


	for record in fund_jensen:
		code    = record[0]
		measure = indicator[code]
		jensen  = measure['jensen']
		ppw     = measure['ppw']
		sortino = measure['sortino']
		stability = measure['stability']
		sharpe    = measure['sharpe']

		largecap = 0
		if code in set(fund_tags['largecap']):
			largecap = 1
		smallcap = 0
		if code in set(fund_tags['smallcap']):
			smallcap = 1
		growth = 0
		if code in set(fund_tags['growthfitness']):
			growth = 1
		value = 0
		if code in set(fund_tags['valuefitness']):
			value = 1
		rise = 0
		if code in set(fund_tags['risefitness']):
			rise = 1
		oscillation = 0
		if code in set(fund_tags['oscillationfitness']):
			oscillation = 1
		decline = 0
		if code in set(fund_tags['declinefitness']):
			decline = 1

		sql = base_sql % (end_date, lookback, 1, code, jensen, ppw, stability, sortino, sharpe, largecap, smallcap, growth, value, rise, oscillation, decline,datetime.now(), datetime.now())
		cursor.execute(sql)


	conn.commit()
	conn.close()

	
	return 1



def bond_fund_measure(allocationdata, start_date, end_date):


	base_sql = "replace into fund_pool (fm_date, fm_look_back, fm_fund_type, fm_fund_code, fm_jensen, fm_ppw, fm_stability, fm_sortino, fm_sharpe, fm_ratebond, fm_creditbond, fm_convertiblebond, created_at, updated_at) values ('%s',%d, %d, '%s',%f, %f, %f, %f, %f, %d, %d, %d, '%s', '%s')"
	
	conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='asset_allocation', charset='utf8')
        cursor = conn.cursor()

	lookback  = 52

	index_df   = DBData.index_value(start_date, end_date)
	bond_df    = DBData.bond_fund_value(start_date, end_date)
		
	codes, indicator     = FundFilter.bondfundfilter(allocationdata, bond_df, index_df[Const.csibondindex_code])
	fund_pool, fund_tags = ST.tagbondfund(allocationdata, bond_df[codes], index_df)

	fund_jensen          = FundIndicator.fund_jensen(bond_df[fund_pool], index_df[Const.csibondindex_code])

	for record in fund_jensen:

		code    = record[0]
		measure = indicator[code]
		jensen  = measure['jensen']
		ppw     = measure['ppw']
		sortino = measure['sortino']
		stability = measure['stability']
		sharpe    = measure['sharpe']

		ratebond = 0
		if code in set(fund_tags['ratebond']):
			ratebond = 1
		creditbond = 0
		if code in set(fund_tags['creditbond']):
			creditbond = 1
		convertiblebond = 0
		if code in set(fund_tags['convertiblebond']):
			convertiblebond = 1

		sql = base_sql % (end_date, lookback, 2, code, jensen, ppw, stability, sortino, sharpe, ratebond, creditbond, convertiblebond ,datetime.now(), datetime.now())
		cursor.execute(sql)


	conn.commit()
	conn.close()	

	return 1



def money_fund_measure(allocationdata, start_date, end_date):


	base_sql = "replace into fund_pool (fm_date, fm_look_back, fm_fund_type, fm_fund_code, fm_sharpe, created_at, updated_at) values ('%s',%d, %d, '%s', %f ,'%s', '%s')"

	conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='asset_allocation', charset='utf8')
        cursor = conn.cursor()

	lookback  = 52

        money_df      = DBData.money_fund_value(start_date, end_date)
	fund_sharpe   = FundIndicator.fund_sharp_annual(money_df)

	for record in fund_sharpe:
		code   = record[0]
		sharpe = record[1]	
		if sharpe < 0:
			continue
		sql = base_sql % (end_date, lookback, 3, code, sharpe ,datetime.now(), datetime.now())
		cursor.execute(sql)

	conn.commit()
	conn.close()

	return 1


def other_fund_measure(allocationdata, start_date, end_date):


	base_sql = "replace into fund_pool (fm_date, fm_look_back, fm_fund_type, fm_fund_code, fm_sharpe, created_at, updated_at) values ('%s',%d, %d, '%s', %f ,'%s', '%s')"

	conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='asset_allocation', charset='utf8')
        cursor = conn.cursor()

	lookback  = 52

        other_df      = DBData.other_fund_value(start_date, end_date)
	fund_sharpe   = FundIndicator.fund_sharp_annual(other_df)


	for record in fund_sharpe:
		code   = record[0]
		sharpe = record[1]	

		if code == 'GLNC':
			sql = base_sql % (end_date, lookback, 4, '159937', sharpe ,datetime.now(), datetime.now())
			cursor.execute(sql)
		if code == 'SP500.SPI':
			sql = base_sql % (end_date, lookback, 4, '513500', sharpe ,datetime.now(), datetime.now())
			cursor.execute(sql)
		if code == 'HSCI.HI':
			sql = base_sql % (end_date, lookback, 4, '513600', sharpe ,datetime.now(), datetime.now())
			cursor.execute(sql)
		
	
	conn.commit()
	conn.close()		

	return 1



if __name__ == '__main__':

	lookback = 52
	dates = DBData.all_trade_dates()

	start_date = dates[-1 * lookback]	

	allocationdata = AllocationData.allocationdata()
	last_friday = DFUtil.last_friday()
	stock_fund_measure(allocationdata, start_date, last_friday)
	bond_fund_measure(allocationdata, start_date, last_friday)
	money_fund_measure(allocationdata, start_date, last_friday)
	other_fund_measure(allocationdata, start_date, last_friday)

	
