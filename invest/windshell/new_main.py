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


def fundfilter(start_date, end_date):

	indicator = {}

	funddf = data.fund_value(start_date, end_date)
	#print 'hehe'
	#print funddf['000457.OF']
	#print funddf['163001.OF'].to_csv('./tmp/163001.csv')
	indexdf = data.index_value(start_date, end_date, '000300.SH')

	#按照规模过滤
	scale_data     = sf.scalefilter(2.0 / 3)	
	#scale_data     = sf.scalefilter(1.0)	
	#print scale_data
	#按照基金创立时间过滤
	setuptime_data = sf.fundsetuptimefilter(funddf.columns, start_date, data.establish_data())

	#print setuptime_data
	#按照jensen测度过滤
	jensen_data    = sf.jensenfilter(funddf, indexdf, rf, 0.5)
	#jensen_data    = sf.jensenfilter(funddf, indexdf, rf, 1.0)

	#按照索提诺比率过滤
	sortino_data   = sf.sortinofilter(funddf, rf, 0.5)
	#sortino_data   = sf.sortinofilter(funddf, rf, 1.0)

	#按照ppw测度过滤
	ppw_data       = sf.ppwfilter(funddf, indexdf, rf, 0.5)
	#ppw_data       = sf.ppwfilter(funddf, indexdf, rf, 1.0)
	#print ppw_data

	stability_data = sf.stabilityfilter(funddf, 2.0 / 3)
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

	#print 'jensen', '000457.OF' in jensen_set
	#print 'sortino', '000457.OF' in sortino_set
	#print 'ppw', '000457.OF' in ppw_set
	#print 'stability', '000457.OF' in stability_set

	codes = []

	for code in scale_set:
		if (code in setuptime_set) and (code in jensen_set) and (code in sortino_set) and (code in ppw_set) and (code in stability_set):
			codes.append(code)

	for code in codes:
		ind = indicator.setdefault(code, {})
		ind['sharpe']    = sharpe_dict[code]
		ind['jensen']    = jensen_dict[code]	
		ind['sortino']   = sortino_dict[code]
		ind['ppw']	 = ppw_dict[code]
		ind['stability'] = stability_dict[code]


	'''
	indicator_str = "%s,%f,%f,%f,%f,%f\n"
	f = open('./tmp/indicator.csv','w')
	f.write("code,sharpe,jensen,sortino,ppw,stability\n")
	for code in codes:
		f.write(indicator_str % (code, sharpe_dict[code],jensen_dict[code], sortino_dict[code], ppw_dict[code], stability_dict[code]))
		#print code,jensen_dict[code], sortino_dict[code], ppw_dict[code], stability_dict[code]		

	f.flush()	
	f.close()
	'''

	#按照业绩持续性过滤
	#stability_data = sf.stabilityfilter(funddf[codes], 2.0 / 3)
	#print stability_data

	#codes = []	
	#for k, v in stability_data:
	#	codes.append(k)	

	return codes, indicator

if __name__ == '__main__':


	max_drawdown = 0.2

	start_date = '2007-01-05'
	end_date   = '2016-04-22'

	indexdf    =  data.index_value(start_date, end_date, '000300.SH')
	dates      = indexdf.index


	allfunddf  = data.funds()
	allfunddfr = allfunddf.pct_change().fillna(0.0)


	'''
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
	portfolio_risk_control_position = 1.0
	portfolio_vs = []
	portfolio_vs.append(1)

	fundws = {}
	fund_values = {}
	fund_codes  = []
	change_position_index = 0

	net_value_f = open('./tmp/net_value.csv','w')
	net_value_f.write('date, net_value\n')

	for i in range(156, len(dates)):

		if (i - 156) % 13 == 0:

			#change_position_index  = i
			start_date             = dates[i - 52].strftime('%Y-%m-%d')
			allocation_start_date  = dates[i - 13].strftime('%Y-%m-%d')
			end_date               = dates[i].strftime('%Y-%m-%d')
			future_end_date        = dates[-1].strftime('%Y-%m-%d')

			if i + 13 >= len(dates):
				future_end_date = dates[-1].strftime('%Y-%m-%d')
			else:
				future_end_date = dates[i + 13].strftime('%Y-%m-%d')


			codes, indicator       = fundfilter(start_date, end_date)
			fund_pool, fund_tags   = st.tagfunds(start_date, end_date, codes)
			allocation_funddf      = data.fund_value(allocation_start_date, end_date)[fund_pool]
			fund_codes             = fs.select_fund(allocation_funddf, fund_tags)

			#fund_codes = list(fund_pool)

			#print fund_pool
			tags = {}
			for key in fund_tags.keys():
					cs = fund_tags[key]
					for c in cs:
							ts = tags.setdefault(c,[])
							ts.append(key)

			#fund_codes = list(tmp_codes)

			#all_funddf        = data.fund_value(start_date, end_date)

			#fund_codes        = list(indicator.keys())
			#allocation_funddf = data.fund_value(allocation_start_date, end_date)
			#allocation_funddf = allocation_funddf[fund_codes]


			####################################################################
			future_funddf   = data.fund_value(end_date, future_end_date)
			future_funddf   = future_funddf[fund_pool]
			future_codes    = []
			future_funddf_sharp = fi.fund_sharp_annual(future_funddf)
			end_n     = (int)(len(future_funddf_sharp) * 0.25)
			start_n   = (int)(len(future_funddf_sharp) * 0.125)
			for n in range(start_n, end_n):
				future_codes.append(future_funddf_sharp[n][0])
			#fund_codes = future_codes
			#####################################################################


			return_data    = fi.fund_return(allocation_funddf)
			risk_data      = fi.fund_risk(allocation_funddf)
			return_dict = {}
			for k,v in return_data:
				return_dict[k] = v
			risk_dict = {}
			for k,v in risk_data:
				risk_dict[k] = v

			tag_indicator = {}
			for key in tags.keys():
                                #tag_str = str(key) + ","
                                ts = tags[key]
                                ts = set(ts)
                                inds = tag_indicator.setdefault(key, [])
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


			'''
			indicator_str = "%s,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d,%d,%d,%d,%d\n"
			f = open('./tmp/indicator_' + end_date + '.csv','w')
			f.write("code,sharpe,jensen,sortino,ppw,stability,risk,return,rise,oscillation,decline,largecap,smallcap,growth,value\n")
			for code in fund_codes:
				ind = indicator[code]
				tag_ind = tag_indicator.setdefault(code,[-1, -1, -1, -1, -1, -1, -1])
				f.write(indicator_str % (code, ind['sharpe'], ind['jensen'], ind['sortino'], ind['ppw'], ind['stability'], risk_dict[code], return_dict[code], tag_ind[0], tag_ind[1], tag_ind[2], tag_ind[3], tag_ind[4], tag_ind[5], tag_ind[6]))
				#print code,jensen_dict[code], sortino_dict[code], ppw_dict[code], stability_dict[code]		
			f.flush()
			f.close()
			'''

			'''	
			fund_codes, fund_tags  = st.tagfunds(start_date, end_date, codes)

			P = [[-1, 1]]
            Q = [[0.0005]]

            largecap_fund, smallcap_fund = pf.largesmallcapfunds(fund_tags)
            fund_codes, ws = pf.asset_allocation(allocation_start_date, end_date, largecap_fund, smallcap_fund, P, Q)


			#print len(fund_codes)
			fundws = {}

			for n in range(0, len(fund_codes)):
				fundws[fund_codes[n]] = ws[n]

			'''

			ws = []
			for n in range(0, len(fund_codes)):
				ws.append(1.0 / len(fund_codes))

			#print fund_codes
			#print ws


			#print fundws
			last_pv = portfolio_vs[-1]
			fund_values = {}
			for n in range(0, len(fund_codes)):
				fund_values[n] = [last_pv * ws[n]]

			fund_risk_control_date = {}
			fund_risk_control_position = {}


###############################################################################################################

		'''
		current_date = dates[i]
		for code in fund_risk_control_date.keys():
			date = fund_risk_control_date[code]
			if current_date - date > datetime.timedelta(days=30):
				fund_risk_control_position[code] = 1.0

		if portfolio_risk_control_date != None:
			if current_date - portfolio_risk_control_date > datetime.timedelta(days=30):
				portfolio_risk_contorl_position = 1.0

		#风控
		start_date          = dates[i - 53].strftime('%Y-%m-%d')
		end_date            = dates[i - 1].strftime('%Y-%m-%d')

		historydf           = allfunddf[fund_codes]	
		historydf           = historydf[start_date : end_date]

		#his_max_semivariance= fi.fund_maxsemivariance(historydf)
		his_weekly_return   = fi.fund_weekly_return(historydf)
		#his_month_return    = fi.fund_month_return(historydf)


		current_weekly_return = {}
		for code in fund_codes:
			current_weekly_return[code] = allfunddfr.loc[end_date, code]


		for code in current_weekly_return:
			rs = his_weekly_return[code]
			rs.sort()
			if current_weekly_return[code] <= rs[( int )( 0.15 * len(rs) )]:
				fund_risk_control_date[code]          = dates[i-1]
				fund_risk_control_position[code]      = 0.6
			if current_weekly_return[code] <= rs[( int )( 0.05 * len(rs) )]:
                                fund_risk_control_date[code]          = dates[i-1]
                                fund_risk_control_position[code]      = 0.2


		max_pv     = max(portfolio_vs)
		current_pv = portfolio_vs[-1]

		if (max_pv - current_pv) / max_pv >= max_drawdown * 0.9:
			portfolio_risk_control_position = 0.25
			portfolio_risk_control_date = dates[i-1]
		elif (max_pv - current_pv) / max_pv >= max_drawdown * 0.7:
			portfolio_risk_control_position = 0.5
			portfolio_risk_control_date = dates[i-1] 
				
		'''

##################################################################################################################################

		pv    = 0
		d     = dates[i]
		for n in range(0, len(fund_codes)):
			vs = fund_values[n]
			#vs.append(vs[-1]  + vs[-1] * allfunddfr.loc[d, code] * fund_risk_control_position.setdefault(code, 1.0) )
			#vs.append(vs[-1]  + vs[-1] * allfunddfr.loc[d, code] * portfolio_risk_control_position)
			vs.append(vs[-1]  + vs[-1] * allfunddfr.loc[d, fund_codes[n]])
			pv = pv + vs[-1]

		portfolio_vs.append(pv)
		net_value_f.write(str(d) + "," + str(pv) + "\n")
		print d, pv

	net_value_f.flush()
	net_value_f.close()
