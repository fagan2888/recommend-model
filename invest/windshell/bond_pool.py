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


rf = const.rf


def bondfilter(start_date, end_date):

	funddf = data.bond_value(start_date, end_date)

	indexdf = data.bond_index_value(start_date, end_date, 'H11001.CSI')

	#按照规模过滤
	#scale_data     = sf.scalefilter(2.0 / 3)

	#按照基金创立时间过滤
	setuptime_data = sf.fundsetuptimefilter(funddf.columns, start_date, data.bond_establish_data())

	#按照jensen测度过滤
	jensen_data    = sf.jensenfilter(funddf, indexdf, rf, 0.5)

	#按照索提诺比率过滤
	sortino_data   = sf.sortinofilter(funddf, rf, 0.5)

	#按照ppw测度过滤
	ppw_data       = sf.ppwfilter(funddf, indexdf, rf, 0.5)
	#print ppw_data

	stability_data = sf.stabilityfilter(funddf, 2.0 / 3)
	#print stability_data


	'''
	scale_set = set()
	for k, v in scale_data:
		scale_set.add(k)
	'''

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


	for code in setuptime_set:
		if (code in jensen_set) and (code in sortino_set) and (code in ppw_set) and (code in stability_set):
			codes.append(code)


	#按照业绩持续性过滤
	#stability_data = sf.stabilityfilter(funddf[codes], 2.0 / 3)
	#print stability_data

	#codes = []
	#for k, v in stability_data:
	#	codes.append(k)


	return codes


if __name__ == '__main__':

	start_date = '2007-01-05'
	end_date = '2016-04-22'

	bonddf = data.bonds()

	#print type(bonddf.values[0][0])

	bonddfr = bonddf.pct_change().fillna(0.0)

	#print bonddf

	indexdf = data.bond_index_value(start_date, end_date, const.csibondindex_code)
	dates = indexdf.index

	tag = {}
	ratebond        =  ''
	creditbond      =  ''
	convertiblebond =  ''

	for i in range(156, len(dates)):

		if (i - 156) % 13 == 0:

			start_date = dates[i - 52].strftime('%Y-%m-%d')
			allocation_start_date = dates[i - 13].strftime('%Y-%m-%d')
			end_date = dates[i].strftime('%Y-%m-%d')

			codes    = bondfilter(start_date, end_date)
			#print codes
			bond_pool, bond_tags = st.tagbonds(start_date, end_date, codes)

			allocation_funddf = data.bond_value(allocation_start_date, end_date)[bond_pool]
			fund_codes, tag = fs.select_bond(allocation_funddf, bond_tags)

			ratebond = tag.setdefault('ratebond', ratebond)
			creditbond = tag.setdefault('creditbond', creditbond)
			convertiblebond = tag.setdefault('convertiblebond', convertiblebond)

			#print tag
		#print tag
		# print fund_codes

		d = dates[i]
		print d.strftime('%Y-%m-%d'), ',', bonddfr.loc[d, tag.setdefault('ratebond', ratebond)], ',', bonddfr.loc[d, tag.setdefault('creditbond',creditbond )], ',', \
			bonddfr.loc[d, tag.setdefault('convertiblebond', convertiblebond)]


# print tag
# allocation_funddf      = allocation_funddf[fund_codes]


