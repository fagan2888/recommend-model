#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import Financial as fin
import data
from numpy import *
from datetime import datetime



#按照基金成立时间过滤
def fundsetuptimefilter(codes, start_date, indicator_df):

        establish_date_code = set()
        for code in indicator_df.index:
                date = indicator_df['establish_date'][code]
                if date <= datetime.strptime(start_date, '%Y-%m-%d'):
                        establish_date_code.add(code)

	final_codes = []
	for code in codes:
		if code in establish_date_code:
			final_codes.append(code)

	return final_codes	



#按照jensen测度过滤	
def jensenfilter(funddf, indexdf, rf, ratio):

	funddfr = funddf.pct_change()
	indexdfr = indexdf.pct_change()

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

		#print p	
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

	
	funddfr = funddf.pct_change()
	indexdfr = funddf.pct_change()

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

	tran_index = []
	for i in range(0, length):
		if i % 4 == 0:
			tran_index.append(i)

	funddf = funddf.iloc[tran_index]
	funddfr = funddf.pct_change()

	indexdf = indexdf.iloc[tran_index]	
	indexdfr = indexdf.pct_change()
	
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

	fundstab = {}
	fundscore = {}
	length = len(funddfr)


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
				score.append(lastscore + (lastrank - rank))
					
			
	final_fund_stability = {}
	for k, v in fundscore.items():
		final_fund_stability[k] = np.sum(v)
				


	x = final_fund_stability
	sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
	sorted_stability = sorted_x				

	result = []
	for i in range(0, (int)(len(sorted_stability) * ratio)):
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
			if v >= 10000000000:
				continue
	
		scale[code] = v


	x = scale
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=False)
        sorted_scale = sorted_x


        result = []
        for i in range(0, (int)(len(sorted_scale) * ratio)):
                result.append(sorted_scale[i])

	return result	


