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


rf = const.rf


def fundfilter(start_date, end_date):


	funddf = data.fund_value(start_date, end_date)
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

	#按照业绩持续性过滤
	stability_data = sf.stabilityfilter(funddf, 2.0 / 3)
	#print stability_data

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
						

'''
#基金标签
def largesmallcap(codes, start_date, end_date):


	funddf = data.fund_value(start_date, end_date)
	funddf = funddf[codes]


	#大小盘适应度标签
	capindexdf = data.index_value(start_date, end_date, ['399314.SZ', '399316.SZ'])		
	largecaptag = st.largecaptag(funddf, capindexdf)
	smallcaptag = st.smallcaptag(funddf, capindexdf)


	return largecaptag, smallcaptag


def portfolio_weight(train_start, train_end, index_codes ,largecaptag, smallcaptag):

	
	funddf = data.fund_value(train_start, train_end)

	capindexdf = data.index_value(train_start, train_end, ['399314.SZ', '399316.SZ'])		

	fund_rank = []

	largesharp = {}
	largecodes = []
	smallsharp = {}
	smallcodes = []


	for k,v in largecaptag.items():
		largesharp[k] = fin.sharp(v, rf)	

	
	for k,v in smallcaptag.items():
		smallsharp[k] = fin.sharp(v, rf)	


	x = largesharp
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_largesharp = sorted_x
	for k, v in sorted_largesharp:
		largecodes.append(k)	


	x = smallsharp
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_smallsharp = sorted_x
	for k, v in sorted_smallsharp:
		smallcodes.append(k)	


	fund_rank.append(largecodes)
	fund_rank.append(smallcodes)


	ws  = pf.portfolio(capindexdf, funddf,  fund_rank)		
	
	return ws
'''


if __name__ == '__main__':



	train_start_date = ['2007-01-05', '2008-01-04', '2009-01-09', '2010-01-08', '2011-01-07', '2012-01-06']	
	train_end_date   = ['2009-12-31', '2010-12-31', '2011-12-31', '2012-12-31', '2013-12-31', '2014-12-31']
	test_start_date  = ['2010-01-08', '2011-01-07', '2012-01-06', '2013-01-04', '2014-01-03', '2015-01-09']
	test_end_date    = ['2010-12-31', '2011-12-30', '2012-12-31', '2013-12-31', '2014-12-31', '2015-12-31']


	hs300_code               = '000300.SH' #沪深300
	zz500_code               = '000905.SH' #中证500
	largecap_code            = '399314.SZ' #巨潮大盘
	samllcap_code            = '399316.SZ' #巨潮小盘
	largecapgrowth_code      = '399372.SZ' #巨潮大盘成长
	largecapvalue_code       = '399373.SZ' #巨潮大盘价值	
	smallcapgrowth_code      = '399376.SZ' #巨潮小盘成长
	smallcapvalue_code       = '399377.SZ' #巨潮小盘价值


	#for i in range(0 ,len(train_start_date)):
	i = 5
	#for i in range(3 ,4):



	train_start = train_start_date[i]
	train_end   = train_end_date[i]
	test_start  = test_start_date[i]
	test_end    = test_end_date[i]


	#testdf = data.fund_value(test_start, test_end)

	#print train_start, train_end		


	codes                  =   fundfilter(train_start, train_end)			
	
	fund_codes, fund_tags  =   st.tagfunds(train_start, train_end, codes)

	funddf                 =   data.fund_value(train_start, train_end)

	trainfunddf            =   funddf[fund_codes]		

	#print fund_tags

	#bound = []
	#bound.append([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
	#bound.append([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])


	risk, returns, ws, sharp = pf.markowitz(trainfunddf, None)


	P = [[-1, 1]]	
	Q = [[0.0005]]


	pf.asset_allocation(train_start, train_end, fund_tags, P, Q)


	#print risk
	#print returns
	#print sharp

	#print ws
	#print fund_codes



	fundws = {}
	for i in range(0, len(fund_codes)):
		fundws[fund_codes[i]] = ws[i] 


	fund_risk_control_date = {}
	fund_risk_control_position = {}
	portfolio_risk_control_date = None 
	portfolio_risk_contorl_position = 1.0
	portfolio_vs = []


	#print fund_values		


	funddf  = data.fund_value(train_start, test_end)
	funddf  = funddf[fund_codes]
	funddfr = funddf.pct_change()
	funddfr = funddfr.dropna()


	testfunddf    = data.fund_value(test_start, test_end)
	testfunddf    = testfunddf[fund_codes]
	testfunddfr   = testfunddf.pct_change()
	testfunddfr   = testfunddfr.dropna()


	dates = testfunddfr.index


	his_pvs = []
	his_fvs = {}
	his_pvs.append(1)
	for code in fundws.keys():
		his_fvs[code] = [fundws[code]]	
	

	his_funddfr = data.fund_value(train_start, train_end).pct_change().dropna()
	his_index = his_funddfr.index
	for index in his_index:
		pv = 0
		for code in fund_codes:
			r = his_funddfr.loc[index, code]
			vs = his_fvs[code]
			last_his_v = vs[len(vs) - 1]
			vs.append(last_his_v * (1 + r))
			pv = pv + last_his_v * (1 + r)	
		his_pvs.append(pv)

	#print his_pvs

	
	portfolio_vs = []
	portfolio_vs.append(1)


	fund_values = {}
	for i in range(0, len(fund_codes)):
		fund_values[fund_codes[i]] = [ws[i]]


	for i in range(1, len(dates)):


		indicator_end_date  = dates[i - 1]
		current_date        = dates[i]


		historydf           = funddf[train_start : indicator_end_date]
		his_max_semivariance= fi.fund_maxsemivariance(historydf)
		his_weekly_return   = fi.fund_weekly_return(historydf)
		his_month_return    = fi.fund_month_return(historydf)


		currentdf             = funddf[train_start : dates[i]]
		current_semivariance  = fi.fund_semivariance(currentdf)

		current_weekly_return = {}
		for code in fund_codes:
			current_weekly_return[code] = funddfr.loc[current_date, code]
		current_month_return = {}	


		for code in currentdf.columns:
			vs      = currentdf[code]
			length  = len(vs)
			current_month_return[code]  = vs[length - 1] /  vs[length - 5] - 1	

		#print 'his_max_semivariance' , his_max_semivariance
		#print
		#print 'his_weekly_return', his_weekly_return
		#print
		#print 'his_month_return', his_month_return 
		#print
		#print 'current_semivariance', current_semivariance
		#print
		#print 'current_weekly_return', current_weekly_return
		#print
		#print 'current_month_return',current_month_return 		
		#print


		for code in fund_codes:
			if not fund_risk_control_date.has_key(code):
				fund_risk_control_position[code] = 1.0
				continue
			else:
				date = fund_risk_control_date[code]
				if current_date - date > datetime.timedelta(days=30):
					fund_risk_control_position[code] = 1.0


		for code in fund_codes:
			fvs = fund_values[code]
			lastv = fvs[len(fvs) - 1]
			r   = funddfr.loc[current_date ,code]
			#lastv = lastv * ( 1 + r )
			lastv = lastv + lastv *  r * fund_risk_control_position[code]
			fvs.append(lastv)


		pv = 0
		for code in fund_codes:
			fvs = fund_values[code]
			lastv = fvs[len(fvs) - 1]
			pv = pv + lastv
		portfolio_vs.append(pv)


		for code in his_max_semivariance:
			if current_semivariance[code] > his_max_semivariance[code]:
				#print 'semivariance', code, dates[i]			
				fund_risk_control_date[code]          = current_date
				fund_risk_control_position[code]      = 0.6	
	
		for code in current_weekly_return:
			rs = his_weekly_return[code]
			if current_weekly_return[code] <= rs[( int )( 0.15 * len(rs) )]:
				#print 'weekly_return', code, dates[i], current_weekly_return[code], rs[( int )( 0.15 * len(rs) )]		
				fund_risk_control_date[code]          = current_date
				fund_risk_control_position[code]      = 0.6					
			if current_weekly_return[code] <= rs[( int )( 0.05 * len(rs) )]:
				#print 'weekly_return', code, dates[i], current_weekly_return[code], rs[( int )( 0.15 * len(rs) )]		
				fund_risk_control_date[code]          = current_date
				fund_risk_control_position[code]      = 0.2

		#p_maxdrawdown = fi.portfolio_maxdrawdown(pvs)


		#print current_date, fund_risk_control_position
		#print 


		'''
		print current_date, fund_risk_control_position
		print 

		for code in fund_codes:
			fvs = fund_values[code]
			lastv = fvs[len(fvs) - 1]
			r   = funddfr.loc[current_date ,code]
			#lastv = lastv * ( 1 + r )
			lastv = lastv + lastv *  r * fund_risk_control_position[code]
			fvs.append(lastv)


		pv = 0
		for code in fund_codes:
			fvs = fund_values[code]
			lastv = fvs[len(fvs) - 1]
			pv = pv + lastv
		portfolio_vs.append(pv)
		'''

	print portfolio_vs


		#for code in current_month_return:
		#	rs = his_month_return[code]
		#	if current_month_return[code] <= rs[( int )( 0.05 * len(rs) )]:
		#		print 'month_return', code, dates[i], current_month_return[code], rs[( int )( 0.05 * len(rs) )]	


		#for code in fund_codes:
				
		#if i % 13 == 0:
		#	risk, returns, ws, sharp = pf.markowitz(funddf[train_start : indicator_end_date])
										
		#print historydfr 
		#end_date = dates[i]
		#print end_date								


	'''	
	largecaptag, smallcaptag = largesmallcap(codes, train_start, train_end)

	ws = portfolio_weight(train_start, train_end, ['399314.SZ', '399316.SZ'], largecaptag, smallcaptag)


	#print ws	

	pvs, prs = pf.portfolio_value(testdf, ws)

	print train_start, train_end
	print test_start
	print ws

	#print pvs
	print 'sharp : ', fi.portfolio_sharp(prs), 'return : ', fi.portfolio_return(prs), 'risk : ', fi.portfolio_risk(prs)		

	fundsharp = fi.fund_sharp(testdf)
	fundreturn = fi.fund_return(testdf)
	fundrisk   = fi.fund_risk(testdf)

	print 'middle sharp : ', fundsharp[len(fundsharp) / 2][1], 'middle return : ' , fundreturn[len(fundreturn) / 2][1], 'middle risk : ' ,  fundrisk[len(fundrisk) / 2][1]
	#print largecaptag
	#print smallcaptag
	#print codes 

	'''


