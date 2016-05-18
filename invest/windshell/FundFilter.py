#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import Financial as fin
import data
from numpy import *
from datetime import datetime
import const
import fundindicator as fi
import pandas as pd


rf = const.rf


#按照基金成立时间过滤
def fundsetuptimefilter(codes, start_date, indicator_df):

	establish_date_code = set()

	for code in indicator_df.index:

			date = indicator_df['establish_date'][code]
			if datetime.strptime(date,'%Y-%m-%d') <= datetime.strptime(start_date, '%Y-%m-%d'):
					establish_date_code.add(code)

	final_codes = []

	for code in codes:

		if code in establish_date_code:

			final_codes.append(code)

	return final_codes



#按照jensen测度过滤
def jensenfilter(funddf, indexdf, rf, ratio):

	funddfr = funddf.pct_change().fillna(0.0)
	indexdfr = indexdf.pct_change().fillna(0.0)


	jensen = {}
	cols = funddfr.columns
	for col in cols:
		p = []
		m = []
		rs = funddfr[col].values
		#print col, rs
		indexrs = indexdfr.values
		for i in range(0, len(rs)):
			if isnan(rs[i]):
				continue
			else:
				p.append(rs[i])
				m.append(indexrs[i])


		jensen[col] = fin.jensen(p, m, rf)


	x = jensen
	sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
	sorted_jensen = sorted_x

	result = []
	for i in range(0, (int)(len(sorted_jensen) * ratio)):
		result.append(sorted_jensen[i])

	return result



#按照sortino测度过滤
def sortinofilter(funddf, rf, ratio):


	funddfr = funddf.pct_change().fillna(0.0)
	indexdfr = funddf.pct_change().fillna(0.0)

	sortino = {}
	cols = funddfr.columns
	for col in cols:
		p = []
		rs = funddfr[col].values
		for i in range(0, len(rs)):
			if isnan(rs[i]):
				continue
			else:
				p.append(rs[i])

		sortino[col] = fin.sortino(p, rf)


	x = sortino
	sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
	sorted_sortino = sorted_x


	result = []
	for i in range(0, (int)(len(sorted_sortino) * ratio)):
		result.append(sorted_sortino[i])


	return result



#按照ppw测度过滤
def ppwfilter(funddf, indexdf, rf, ratio):


	length = len(funddf.index)

	'''
	tran_index = []
	for i in range(0, length):
		if i % 4 == 0:
			tran_index.append(i)

	funddf = funddf.iloc[tran_index]
	funddfr = funddf.pct_change()

	indexdf = indexdf.iloc[tran_index]
	indexdfr = indexdf.pct_change()
	'''


	funddfr = funddf.pct_change().fillna(0.0)
	indexdfr = indexdf.pct_change().fillna(0.0)

	ppw = {}
	cols = funddfr.columns
	for col in cols:
		p = []
		m = []
		rs = funddfr[col].values
		indexrs = indexdfr.values
		for i in range(0, len(rs)):
			if isnan(rs[i]):
				continue
			else:
				p.append(rs[i] - rf)
				m.append(indexrs[i] - rf)


		ppw[col] = fin.ppw(p, m)


	x = ppw
	sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
	sorted_ppw = sorted_x


	result = []
	for i in range(0, (int)(len(sorted_ppw) * ratio)):
		result.append(sorted_ppw[i])

	#print result
	return result



#基金稳定性测度
def stabilityfilter(funddf, ratio):


	length = len(funddf.index)
	tran_index = []
	for i in range(0, length):
		if i % 4 == 0:
			tran_index.append(i)

	funddf = funddf.iloc[tran_index]
	funddfr = funddf.pct_change()

	length = len(funddfr)

	fundstab = {}
	fundscore = {}

	for i in range(1, length):

		fr = {}
		for code in funddfr.columns:
			r = funddfr[code].values[i]
			fr[code] = r


		x = fr
		sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
		sorted_fr = sorted_x

		l = len(sorted_fr)
		frank = {}


		for i in range(0, l):
			k,v = sorted_fr[i]
			if i <= 0.2 * l:
				frank[k] = 5
			elif i > 0.2 * l and i <= 0.4 * l:
				frank[k] = 4
			elif i > 0.4 * l and i <= 0.6 * l:
				frank[k] = 3
			elif i > 0.6 * l and i <= 0.8 * l:
				frank[k] = 2
			else:
				frank[k] = 1


		for code in frank.keys():
			stab = fundstab.setdefault(code, [])
			score = fundscore.setdefault(code, [])

			rank = frank[code]

			if len(stab) == 0:
				stab.append(rank)
				score.append(0)
				continue

			lastrank = stab[len(stab) - 1]
			lastscore = score[len(score) - 1]


			if rank >= lastrank:
				score.append(5)
			else:
				score.append(lastscore - (lastrank - rank))

			stab.append(rank)


	final_fund_stability = {}
	for k, v in fundscore.items():
		final_fund_stability[k] = np.sum(v)


	x = final_fund_stability
	sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
	sorted_stability = sorted_x

	result = []
	for i in range(0, (int)(len(sorted_stability) * ratio)):
		#print sorted_stability[i]
		result.append(sorted_stability[i])

	return result



#按照规模过滤
def scalefilter(ratio):

	fund_scale_df =  data.scale_data()
	stock_codes   =  data.stock_fund_code()

	scale = {}
	for code in fund_scale_df.index:
		v = fund_scale_df.loc[code].values
		if code in stock_codes:
			#if string.atof(v[0]) >= 10000000000.0:
			continue

		scale[code] = v

	#print 'scale 000457 : ' ,scale['000457.OF']
	x = scale
	sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=False)
	sorted_scale = sorted_x


	result = []
	for i in range(0, (int)(len(sorted_scale) * ratio)):
		result.append(sorted_scale[i])

	return result



def stockfundfilter(start_date, end_date):


	indicator = {}

	funddf = data.fund_value(start_date, end_date)
	#print 'hehe'
	#print funddf['000457.OF']
	#print funddf['163001.OF'].to_csv('./tmp/163001.csv')
	indexdf = data.index_value(start_date, end_date, '000300.SH')


	#按照规模过滤
	scale_data     = scalefilter(2.0 / 3)
	#scale_data     = sf.scalefilter(1.0)
	#print scale_data
	#按照基金创立时间过滤


	setuptime_data = fundsetuptimefilter(funddf.columns, start_date, data.establish_data())


	#print setuptime_data
	#按照jensen测度过滤
	jensen_data    = jensenfilter(funddf, indexdf, rf, 0.5)
	#jensen_data    = sf.jensenfilter(funddf, indexdf, rf, 1.0)

	#按照索提诺比率过滤
	sortino_data   = sortinofilter(funddf, rf, 0.5)
	#sortino_data   = sf.sortinofilter(funddf, rf, 1.0)

	#按照ppw测度过滤
	ppw_data       = ppwfilter(funddf, indexdf, rf, 0.5)
	#ppw_data       = sf.ppwfilter(funddf, indexdf, rf, 1.0)
	#print ppw_data

	stability_data = stabilityfilter(funddf, 2.0 / 3)
	#stability_data = sf.stabilityfilter(funddf, 1.0)

	sharpe_data    = fi.fund_sharp_annual(funddf)

	#print stability_data

	#print 'jensen'
	jensen_dict = {}
	for k,v in jensen_data:
		jensen_dict[k] = v
		#print k, v
	#print
	#print 'sortino'


	sortino_dict = {}
	for k,v in sortino_data:
		sortino_dict[k] = v
		#print k,v

	#print
	#print 'ppw'
	ppw_dict = {}
	for k,v in ppw_data:
		ppw_dict[k] = v
		#print k,v

	#print
	#print 'statbility'
	stability_dict = {}
	for k,v in stability_data:
		stability_dict[k] = v
		#print k,v


	sharpe_dict = {}
	for k,v in sharpe_data:
		sharpe_dict[k] = v


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


	for code in codes:
		ind = indicator.setdefault(code, {})
		ind['sharpe']    = sharpe_dict[code]
		ind['jensen']    = jensen_dict[code]
		ind['sortino']   = sortino_dict[code]
		ind['ppw']	     = ppw_dict[code]
		ind['stability'] = stability_dict[code]


	indicator_codes = []
	indicator_datas = []


	indicator_set = set()

	for code in scale_set:
		if code in setuptime_set:
			indicator_set.add(code)


	for code in indicator_set:
		indicator_codes.append(code)
		indicator_datas.append([sharpe_dict.setdefault(code, None), jensen_dict.setdefault(code, None), sortino_dict.setdefault(code, None), ppw_dict.setdefault(code, None), stability_dict.setdefault(code, None)])



	indicator_df = pd.DataFrame(indicator_datas, index = indicator_codes, columns=['sharpe', 'jensen', 'sortino', 'ppw', 'stability'])
	indicator_df.to_csv('./tmp/stock_indicator_' + end_date + '.csv')

	f = open('./tmp/stockfilter_codes_' + end_date + '.csv','w')

	for code in codes:
		f.write(str(code) + '\n')

	f.flush()
	f.close()

	return codes, indicator



def bondfundfilter(start_date, end_date):

	indicator = {}

	funddf = data.bond_value(start_date, end_date)
	indexdf = data.bond_index_value(start_date, end_date, 'H11001.CSI')

	#按照基金创立时间过滤
	setuptime_data = fundsetuptimefilter(funddf.columns, start_date, data.bond_establish_data())


	#print setuptime_data
	#按照jensen测度过滤
	jensen_data    = jensenfilter(funddf, indexdf, rf, 0.5)
	#jensen_data    = sf.jensenfilter(funddf, indexdf, rf, 1.0)

	#按照索提诺比率过滤
	sortino_data   = sortinofilter(funddf, rf, 0.5)
	#sortino_data   = sf.sortinofilter(funddf, rf, 1.0)

	#按照ppw测度过滤
	ppw_data       = ppwfilter(funddf, indexdf, rf, 0.5)
	#ppw_data       = sf.ppwfilter(funddf, indexdf, rf, 1.0)
	#print ppw_data

	stability_data = stabilityfilter(funddf, 2.0 / 3)
	#stability_data = sf.stabilityfilter(funddf, 1.0)

	sharpe_data    = fi.fund_sharp_annual(funddf)

	#print stability_data

	#print 'jensen'
	jensen_dict = {}
	for k,v in jensen_data:
		jensen_dict[k] = v
	#print k, v
	#print
	#print 'sortino'


	sortino_dict = {}
	for k,v in sortino_data:
		sortino_dict[k] = v
	#print k,v

	#print
	#print 'ppw'
	ppw_dict = {}
	for k,v in ppw_data:
		ppw_dict[k] = v
	#print k,v

	#print
	#print 'statbility'
	stability_dict = {}
	for k,v in stability_data:
		stability_dict[k] = v
	#print k,v


	sharpe_dict = {}
	for k,v in sharpe_data:
		sharpe_dict[k] = v


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


	for code in codes:
		ind = indicator.setdefault(code, {})
		ind['sharpe']    = sharpe_dict[code]
		ind['jensen']    = jensen_dict[code]
		ind['sortino']   = sortino_dict[code]
		ind['ppw']	     = ppw_dict[code]
		ind['stability'] = stability_dict[code]


	indicator_set = set()
	for code in setuptime_set:
		indicator_set.add(code)

	indicator_codes = []
	indicator_datas = []


	for code in indicator_set:
		indicator_codes.append(code)
		indicator_datas.append([sharpe_dict.setdefault(code, None), jensen_dict.setdefault(code, None), sortino_dict.setdefault(code, None), ppw_dict.setdefault(code, None), stability_dict.setdefault(code, None)])


	indicator_df = pd.DataFrame(indicator_datas, index = indicator_codes, columns=['sharpe', 'jensen', 'sortino', 'ppw', 'stability'])
	indicator_df.to_csv('./tmp/bond_indicator_' + end_date + '.csv')


	f = open('./tmp/bondfilter_codes_' + end_date + '.csv','w')


	for code in codes:
		f.write(str(code) + '\n')

	f.flush()
	f.close()


	return codes, indicator






