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
import FundFilter as ff


def fund_maxsemivariance(funddf):


	fundsemivariance = {}
	
	funddfr = funddf.pct_change().fillna(0.0)

	codes = funddfr.columns

	for code in codes:

		rs = []	
		for r in funddfr[code].values:
			if not isnan(r):
				rs.append(r)
		max_semivariance = 0

		for i in range(5, len(rs) + 1):
			semivariance = fin.semivariance(rs[0 : i])
			if semivariance > max_semivariance:
					max_semivariance = semivariance

		fundsemivariance[code] = max_semivariance
	
	return fundsemivariance


def fund_semivariance(funddf):


	fundsemivariance = {}
	
	funddfr = funddf.pct_change().fillna(0.0)

	codes = funddfr.columns


	for code in codes:
		rs = []	
		for r in funddfr[code].values:
			if not isnan(r):
				rs.append(r)

		fundsemivariance[code] = fin.semivariance(rs)
	

	return fundsemivariance


def fund_weekly_return(funddf):
		
	fundweeklyreturn = {}
	
	funddfr = funddf.pct_change().fillna(0.0)

	codes = funddfr.columns

	for code in codes:
		rs = []	
		for r in funddfr[code].values:
			if not isnan(r):
				rs.append(r)
		rs.sort()	
		fundweeklyreturn[code] = rs
	

	return fundweeklyreturn


def fund_month_return(funddf):

	fundmonthreturn = {}

	length = len(funddf.index)


	tran_index = []
	for i in range(0, length):
		if i % 4 == 0:
			tran_index.append(i)

	funddf = funddf.iloc[tran_index]
	
	funddfr = funddf.pct_change().fillna(0.0)

	codes = funddfr.columns

	for code in codes:
		rs = []	
		for r in funddfr[code].values:
			if not isnan(r):
				rs.append(r)
	
		rs.sort()
		fundmonthreturn[code] = rs
	

	return fundmonthreturn


def fund_sharp(funddf):

	fundsharp = {}
	
	funddfr = funddf.pct_change().fillna(0.0)

	codes = funddfr.columns

	for code in codes:
		rs = []	
		for r in funddfr[code].values:
			if not isnan(r):
				rs.append(r)

		fundsharp[code] = fin.sharp(rs, const.rf)
	

	x = fundsharp
	sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
	sorted_sharp = sorted_x

	result = []
	for i in range(0, len(sorted_sharp)):
		result.append(sorted_sharp[i])

	return result


def fund_sharp_annual(funddf):

	fundsharp = {}
	
	funddfr = funddf.pct_change().fillna(0.0)

	codes = funddfr.columns

	for code in codes:
		rs = []	
		for r in funddfr[code].values:
			rs.append(r)

		fundsharp[code] = fin.sharp_annual(rs, Const.rf)

	x = fundsharp
    	sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_sharp = sorted_x

        result = []
        for i in range(0, len(sorted_sharp)):
                result.append(sorted_sharp[i])


	return result


def fund_return(funddf):


	fundreturn = {}
	
	funddfr = funddf.pct_change().fillna(0.0)

	codes = funddfr.columns

	for code in codes:
		vs = funddfr[code].values
		#fundreturn[code] = vs[len(vs) -1] / vs[0] - 1
		fundreturn[code] = np.mean(vs)
		#print code, fundreturn[code]
	
	x = fundreturn
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_return = sorted_x

        result = []
        for i in range(0, len(sorted_return)):
                result.append(sorted_return[i])


	return result



def fund_risk(funddf):


	fundrisk = {}
	
	funddfr = funddf.pct_change().fillna(0.0)

	codes = funddf.columns

	for code in codes:
		fundrisk[code] = np.std(funddfr[code].values)
	
	x = fundrisk
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_risk = sorted_x

        result = []
        for i in range(0, len(sorted_risk)):
                result.append(sorted_risk[i])

	return result



def portfolio_sharpe(pvs):
	rs = []
	for i in range(1, len(pvs)):
		rs.append(pvs[i] / pvs[i-1] - 1)
	returns = np.mean(rs) * 52
	risk    = np.std(rs) * (52 ** 0.5)
	return (returns - 0.03) / risk


def portfolio_return(pvs):
	#print pvs
	#print pvs[0]
	#print pvs[len(pvs) - 1]
	rs = []
	for i in range(1, len(pvs)):
		rs.append(pvs[i] / pvs[i-1] - 1)
	return np.mean(rs) * 52
	#return pvs[len(pvs) - 1] / pvs[0] - 1


def portfolio_risk(pvs):
	rs = []
	for i in range(1, len(pvs)):
		rs.append(pvs[i] / pvs[i-1] - 1)
	return np.std(rs)



def portfolio_maxdrawdown(pvs):
	mdd = 0
	peak = pvs[0]
	for v in pvs:
		if v > peak:
			peak = v
		dd = (peak - v) / peak
		if dd > mdd:
			mdd = dd
	return mdd



#基金的最大回撤
def fund_maxdrawdown(funddf):

	return 0	



if __name__ == '__main__':

	start_date = '2015-04-20'
	end_date   = '2016-04-22'

	funddf     =  data.fund_value(start_date, end_date)
	indexdf = data.index_value(start_date, end_date, '000300.SH')	

	#df = funddf['000398.OF']

	#print np.mean(df.pct_change()) * 52	
	#按照规模过滤
        scale_data     = sf.scalefilter(3.0 / 3)

        #按照基金创立时间过滤
        setuptime_data = sf.fundsetuptimefilter(funddf.columns, start_date, data.establish_data())

        #按照jensen测度过滤
        jensen_data    = sf.jensenfilter(funddf, indexdf, const.rf, 1.0)

        #按照索提诺比率过滤
        sortino_data   = sf.sortinofilter(funddf, const.rf, 1.0)

        #按照ppw测度过滤
        ppw_data       = sf.ppwfilter(funddf, indexdf, const.rf, 1.0)
        #print ppw_data

        stability_data = sf.stabilityfilter(funddf, 3.0 / 3)

        sharpe_data    = fi.fund_sharp_annual(funddf)
	
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



	codes = []
	for code in scale_set:
		if code in setuptime_set:
			codes.append(code)


	#print 'code, sharpe, jensen, sortino, ppw, stability'	
	#for code in codes:
	#	print code, ',', sharpe_dict[code], ',', jensen_dict[code],',', sortino_dict[code] ,',', ppw_dict[code],',' ,stability_dict[code]	

			
