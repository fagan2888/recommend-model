#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import const
import Financial as fin
import stockfilter as sf
import stocktag as st
import portfolio as pf
import fundindicator as fi
import fund_selector as fs
import data
import datetime
from numpy import *
import fund_evaluation as fe
import pandas as pd
from sklearn.cluster import KMeans


rf = const.rf

def fundfilter(start_date, end_date):

	funddf = data.fund_value(start_date, end_date)
	#print funddf
	indexdf = data.index_value(start_date, end_date, '000300.SH')

	#按照规模过滤
	scale_data     = sf.scalefilter(2.0 / 3)	
	
	#按照基金创立时间过滤
	setuptime_data = sf.fundsetuptimefilter(funddf.columns, start_date, data.establish_data())

	#按照jensen测度过滤
	jensen_data    = sf.jensenfilter(funddf, indexdf, rf, 0.5)

	#按照索提诺比率过滤
	sortino_data   = sf.sortinofilter(funddf, rf, 0.5)

	#按照ppw测度过滤
	ppw_data       = sf.ppwfilter(funddf, indexdf, rf, 0.5)
	#print ppw_data

	stability_data = sf.stabilityfilter(funddf, 2.0 / 3)

	scale_set = set()
	for k, v in scale_data:
		scale_set.add(k)

	setuptime_set = set(setuptime_data)

	jensen_set = set()
	for k, v in jensen_data:
		jensen_set.add(k)

	sortino_set = set()
	for k, v in sortino_data:
		sortino_set.add(k)

	ppw_set = set()
	for k, v in ppw_data:
		ppw_set.add(k)

	stability_set = set()
	for k, v in stability_data:
		stability_set.add(k)

	codes = []

	for code in scale_set:
		if (code in setuptime_set) and (code in jensen_set) and (code in sortino_set) and (code in ppw_set) and (code in stability_set):
			codes.append(code)

	return codes


if __name__ == '__main__':

	hs300_code               = '000300.SH' #沪深300
	zz500_code               = '000905.SH' #中证500
	largecap_code            = '399314.SZ' #巨潮大盘
	samllcap_code            = '399316.SZ' #巨潮小盘
	largecapgrowth_code      = '399372.SZ' #巨潮大盘成长
	largecapvalue_code       = '399373.SZ' #巨潮大盘价值	
	smallcapgrowth_code      = '399376.SZ' #巨潮小盘成长
	smallcapvalue_code       = '399377.SZ' #巨潮小盘价值

	start_date = '2007-01-05'
	end_date   = '2016-04-22'

	indexdf    =  data.index_value(start_date, end_date, '000300.SH')
	dates = indexdf.index

	funddf = data.funds()
	funddfr = funddf.pct_change().fillna(0.0)

	largecap_r    = {}
	smallcap_r    = {}
	rise_r        = {}
	oscillation_r = {}
	decline_r     = {}
	growth_r      = {}
	valur_r       = {}

	for i in range(156, len(dates)):

		if (i - 156) % 13 == 0:

			start_date                   = dates[i - 52].strftime('%Y-%m-%d')
			allocation_start_date        = dates[i - 13].strftime('%Y-%m-%d')

			end_date                     = dates[i].strftime('%Y-%m-%d')
			codes                        = fundfilter(start_date, end_date)
			fund_pool, fund_tags         = st.tagfunds(start_date, end_date, codes)

			allocation_funddf            = data.fund_value(allocation_start_date, end_date)[fund_pool]
			fund_codes, tag              = fs.select_fund(allocation_funddf, fund_tags)

			#print tag
			#print fund_codes

		d = dates[i]
		print d.strftime('%Y-%m-%d'), ',' , funddfr.loc[d, tag['largecap']],',', funddfr.loc[d, tag['smallcap']],',',\
				funddfr.loc[d, tag['rise']], ',', funddfr.loc[d, tag['oscillation']], ',', funddfr.loc[d, tag['decline']], ',', \
				funddfr.loc[d, tag['growth']], ',', funddfr.loc[d, tag['value']]
			#print tag
			#allocation_funddf      = allocation_funddf[fund_codes]




