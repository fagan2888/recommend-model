#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import Financial as fin
import const
import data
from numpy import isnan
from datetime import datetime
import pandas as pd


def fund_maxsemivariance(funddf):


	fundsemivariance = {}
	
	funddfr = funddf.pct_change().dropna()

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
	
	funddfr = funddf.pct_change().dropna()

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
	
	funddfr = funddf.pct_change().dropna()

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
	
	funddfr = funddf.pct_change().dropna()

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
	
	funddfr = funddf.pct_change().dropna()

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
	
	funddfr = funddf.pct_change().dropna()

	codes = funddfr.columns

	for code in codes:
		rs = []	
		for r in funddfr[code].values:
			if not isnan(r):
				rs.append(r)

		fundsharp[code] = fin.sharp_annual(rs, const.rf)
	

	x = fundsharp
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_sharp = sorted_x

        result = []
        for i in range(0, len(sorted_sharp)):
                result.append(sorted_sharp[i])


	return result





def fund_return(funddf):


	fundreturn = {}
	
	funddfr = funddf.pct_change().dropna()

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
	
	funddfr = funddf.pct_change().dropna()

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



def portfolio_sharp(prs):
	return fin.sharp(prs, const.rf)



def portfolio_return(prs):
	#print pvs
	#print pvs[0]
	#print pvs[len(pvs) - 1]
	return np.mean(prs)
	#return pvs[len(pvs) - 1] / pvs[0] - 1


def portfolio_risk(prs):
	return np.std(prs)



def portfolio_maxdrawdown(pvs):

	inv_list =  np.array(pvs)
	running_max = pd.expanding_max(inv_list)
	diff = (inv_list - running_max)/running_max

	return diff


#基金的最大回撤
def fund_maxdrawdown(funddf):

	return 0	


