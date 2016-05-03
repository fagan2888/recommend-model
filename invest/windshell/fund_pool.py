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
import data
import datetime
from numpy import *
import fund_evaluation as fe
import pandas as pd


from sklearn.cluster import KMeans


rf = const.rf

indicator = {}

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

	sharpe_data    = fi.fund_sharp(funddf)	


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

	'''
	codes = list(jensen_dict.keys())
	codes.sort()


	jensen_array = []
	sortino_array = []
	ppw_array = []
	stability_array = []

	for code in codes:
		jensen_array.append(jensen_dict[code] if jensen_dict.has_key(code) else 0)
		sortino_array.append(sortino_dict[code] if sortino_dict.has_key(code) else 0)
		ppw_array.append(ppw_dict[code] if ppw_dict.has_key(code) else 0)
		stability_array.append(stability_dict[code] if stability_dict.has_key(code) else 0)


	indicators = {'code':codes, 'jensen':jensen_array, 'sortino':sortino_array, 'ppw':ppw_array,'stability':stability_array}	

	frame = pd.DataFrame(indicators)			

	frame.to_csv('./wind/fund_indicator.csv')
	'''


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

	#indicator_str = "%s,%f,%f,%f,%f,%f\n"	

	for code in codes:
		inds = indicator.setdefault(code, [])
		inds.append(code)
		inds.append(sharpe_dict[code])
		inds.append(jensen_dict[code])
		inds.append(sortino_dict[code])
		inds.append(ppw_dict[code])
		inds.append(stability_dict[code])
		#inds.append(return_dict[code])
		#inds.append(risk_dict[code])


	#f = open('./tmp/indicator.csv','w')
	#f.write("code,sharpe,jensen,sortino,ppw,stability\n")
	#for code in codes:
	#	f.write(indicator_str % (code, sharpe_dict[code],jensen_dict[code], sortino_dict[code], ppw_dict[code], stability_dict[code]))
		#print code,jensen_dict[code], sortino_dict[code], ppw_dict[code], stability_dict[code]		
	#f.flush()	
	#f.close()



	#按照业绩持续性过滤
	#stability_data = sf.stabilityfilter(funddf[codes], 2.0 / 3)
	#print stability_data

	#codes = []	
	#for k, v in stability_data:
	#	codes.append(k)	

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


	'''
	fund_risk_control_date = {}
	fund_risk_control_position = {}
	portfolio_risk_control_date = None 
	portfolio_risk_contorl_position = 1.0
	portfolio_vs = []
	portfolio_vs.append(1)
	'''


	start_date = '2007-01-05'
	end_date   = '2016-04-22'


	indexdf    =  data.index_value(start_date, end_date, '000300.SH')
	dates = indexdf.index


	all_codes = []


	pool_vs = {}
	fund_vs = {}
	selected_fund_vs = {}			
	allocation_vs    = {}
	cluster_vs       = {}


	fundws = {}
	poolws = {}
	selectedfundws = {}
	allocationws   = {}
	clusterws      = {}


	funddf = data.funds()
	funddfr = funddf.pct_change().fillna(0.0)


	#pool_f = open('./tmp/fund_pool.csv','w')


	for i in range(156, len(dates)):
	#for i in range(156, 200):

		if (i - 156) % 13 == 0:

			indicator = {}

			start_date             = dates[i - 52].strftime('%Y-%m-%d')
			pool_start_date        = dates[i - 13].strftime('%Y-%m-%d')

			end_date               = dates[i].strftime('%Y-%m-%d')
			codes                  = fundfilter(start_date, end_date)			
			fund_codes, fund_tags  = st.tagfunds(start_date, end_date, codes)
				
			fund_codes = set()
			for key in fund_tags.keys():
				for code in fund_tags[key]:
					fund_codes.add(code)		


			funddf                 = data.fund_value(start_date, end_date)
			pool_codes             = list(fund_codes)
			
			fund_valuedf           = data.fund_value(start_date, end_date)
			fund_codes             = list(fund_valuedf.columns)


			pool_fund_df = data.fund_value(pool_start_date, end_date)
			pool_fund_df = pool_fund_df[fund_valuedf.columns]


			return_data    = fi.fund_return(funddf)	
			risk_data      = fi.fund_risk(funddf)	
			return_dict = {}
			for k,v in return_data:
				return_dict[k] = v
				ind = indicator.setdefault(k,[])
				ind.append(v)
				#indicator[k].append(v)
			risk_dict = {}
			for k,v in risk_data:
				risk_dict[k] = v
				#indictor[k].append(v)
				ind = indicator.setdefault(k,[])
				ind.append(v)

	
			P = [[-1, 1]]
                	Q = [[0.0005]]

                	largecap_fund, smallcap_fund = pf.largesmallcapfunds(fund_tags)
                	selected_codes, ws = pf.asset_allocation(pool_start_date, end_date, largecap_fund, smallcap_fund, P, Q)

			for n in range(0, len(fund_codes)):
				fundws[fund_codes[n]] = 1.0 / len(fund_codes) 

			for n in range(0, len(pool_codes)):
				poolws[pool_codes[n]] = 1.0 / len(pool_codes) 

			for n in range(0, len(selected_codes)):
				selectedfundws[selected_codes[n]] = 1.0 / len(selected_codes) 

			for n in range(0, len(selected_codes)):
				allocationws[selected_codes[n]] = ws[n] 

			tags = {}
			for key in fund_tags.keys():
				cs = fund_tags[key]
				for c in cs:
					ts = tags.setdefault(c,[])
					ts.append(key)

			#head = "code,rise,oscillation,decline,large,small,growth,value\n"
			#f = open('./tmp/tags.csv','w')
			#f.write(head)
			for key in tags.keys():
				#tag_str = str(key) + ","
				ts = tags[key]
				ts = set(ts)
				inds = indicator.setdefault(key, [])
				if 'risefitness' in ts:
					inds.append(1)
				else:
					inds.append(0)
				if 'oscillationfitness' in ts:
					inds.append(1)
				else:
					inds.append(0)
				if 'declinefitness' in ts:
					inds.append(1)
				else:
					inds.append(0)
				if 'largecap' in ts:
					inds.append(1)
				else:
					inds.append(0)
				if 'smallcap' in ts:
					inds.append(1)
				else:
					inds.append(0)
				if 'growthfitness' in ts:
					inds.append(1)
				else:
					inds.append(0)
				if 'valuefitness' in ts:
					inds.append(1)
				else:
					inds.append(0)

			
			labels = {}
			feature = []
			cluster = {}
			for code in pool_codes:
				feature.append([risk_dict[code], return_dict[code]])	
			clf = KMeans(n_clusters = 3)
			clf.fit(feature)


			for n in range(0, len(pool_codes)):
				labels[pool_codes[n]] = clf.labels_[n]	
				ind = indicator.setdefault(pool_codes[n], [])
				ind.append(clf.labels_[n])
				cl  = cluster.setdefault(clf.labels_[n], [])
				cl.append(pool_codes[n])



			for k,v in cluster.items():
				clfunddf = pool_fund_df[v]
				if len(v) > 1:			
					final_risk, final_return, final_ws,final_sharp = pf.markowitz(clfunddf, None)	
					for m in range(0, len(v)):
						clusterws[v[m]] = final_ws[m] / 3.0
				else:
					c = v[0]
					clusterws[c] = 1.0 / 3.0	

			for code in clusterws.keys():		
				w = clusterws[code]
				if w < 0.02:
					del(clusterws[code])

			sumws = 0
			for code in clusterws.keys():
				sumws = sumws + clusterws[code]


			for code in clusterws.keys():
				clusterws[code] = clusterws[code] / sumws

	
				#print final_ws

			print clusterws


			#print end_date, indicator
			ind_format = "%s,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d,%d,%d,%d,%d,%d\n"
			head       = 'code, sharpe, jensen, sortino, ppw, stability, return, risk, rise, oscillation, decline, largecap, smallcap, growth, value,label\n'
			f = open('./tmp/indicator_tags_' + end_date +'.csv','w')
			f.write(head)	
			for code in tags.keys():
				ind = indicator[code]
				f.write(ind_format % tuple(ind))
			f.flush()
			f.close()


			
			#print clf.cluster_centers_
			#print clf.labels_

			#pool_funddf = data.fund_value(pool_start_date, end_date)
									
			#f.write(tag_str)
			#f.flush()
			#f.close()

			#inds.append(code)
			#inds.append(sharpe_dict[code])
				
		#funddf  = data.fund_value(dates[i - 1].strftime('%Y-%m-%d'), dates[i].strftime('%Y-%m-%d'))
		#funddfr = funddf.pct_change()


		pre_d   = dates[i - 1]
		d       = dates[i]


		pr      = 0.0
		fr      = 0.0
		sr      = 0.0
		ar      = 0.0
		cr      = 0.0


		for code in pool_codes:
			pr = pr + poolws.setdefault(code, 0.0) * funddfr.loc[d, code]	
		v = pool_vs.setdefault(pre_d,1)
		pool_vs[d] = v * ( 1 + pr )	
		#print d, v * ( 1 + pr)

			
		for code in fund_codes:
			fr = fr + fundws.setdefault(code, 0.0) * funddfr.loc[d, code]	
		v = fund_vs.setdefault(pre_d,1)
		fund_vs[d] = v * ( 1 + fr )	
		#print d, v * ( 1 + pr)


		for code in selected_codes:
			sr = sr + selectedfundws.setdefault(code, 0.0) * funddfr.loc[d, code]	
					
			ar = ar + allocationws.setdefault(code, 0.0) * funddfr.loc[d, code]	

		for code in clusterws.keys():
			cr = cr + clusterws.setdefault(code,0.0) * funddfr.loc[d, code]


		v = selected_fund_vs.setdefault(pre_d,1)
		selected_fund_vs[d] = v * ( 1 + sr )	
		#print d, v * ( 1 + pr)

		v = allocation_vs.setdefault(pre_d,1)
		allocation_vs[d] = v * ( 1 + ar )	
		#print d, v * ( 1 + pr)

		v = allocation_vs.setdefault(pre_d,1)
		allocation_vs[d] = v * ( 1 + ar )	


		v = cluster_vs.setdefault(pre_d,1)
		cluster_vs[d] = v * ( 1 + cr )	


		print d, cluster_vs[d], allocation_vs[d], selected_fund_vs[d], pool_vs[d], fund_vs[d]	


	#pool_f.flush()
	#pool_f.close()	

	f = open('./tmp/pool.csv','w')
	head = "date,cluster_fund, allocation_fund, selected_fund, fund_pool, fund\n"	
	f.write(head)
	vformat = "%s,%f,%f,%f,%f,%f\n"

	ds = pool_vs.keys()
	ds = list(ds)
	ds.sort()
	for d in ds:
		poolv = pool_vs[d]
		fundv = fund_vs[d]
		selectedv = selected_fund_vs[d]
		allocationv = allocation_vs[d]
		clusterv    = cluster_vs[d]
		f.write(vformat % (d.strftime('%Y-%m-%d'), clusterv, allocationv, selectedv, poolv, fundv))

	f.flush()
	f.close()
