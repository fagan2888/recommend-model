#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import Financial as fin
import const
import data
from numpy import *
from datetime import datetime


#strategicallocation


#indexallocation


#technicallocation



#中类资产配置
def indexallocation(indexdf):

	indexdfr = indexdf.pct_change()
		
	indexdfr = indexdfr.dropna()	
	
	codes = indexdfr.columns

	return_rate = []
	for code in codes:
		return_rate.append(indexdfr[code].values)

	#print return_rate
	risks, returns, ws = fin.efficient_frontier_index(return_rate)

	rf = const.rf

	final_risk = 0
	final_return = 0
	final_ws = []
	final_sharp = -1000


	for i in range(0, len(risks)):

		
		sharp = (returns[i] - rf) / risks[i]	

		if sharp > final_sharp:

				final_risk = risks[i]
				final_return = returns[i]
				final_ws     = ws[i]
				final_sharp  = sharp			

	return final_risk, final_return, final_ws, final_sharp



#细类资产配置
def technicallocation(funddf, fund_rank):

        rf = const.rf

	funddfr = funddf.pct_change()

	funddfr = funddfr.dropna()

	
	final_risk = 0
	final_return = 0
	final_ws = []
	final_sharp = -1000
	final_codes = []
		
	for i in range(2, min(11, len(fund_rank))):

		codes = fund_rank[0 : i]
		dfr = funddfr[codes]

		dfr.dropna()
		
		return_rate = []
		for code in codes:
			return_rate.append(dfr[code].values)

		#print return_rate	
		risks, returns, ws = fin.efficient_frontier_fund(return_rate)
 

		for j in range(0, len(risks)):

			sharp = (returns[j] - rf) / risks[j]
			if sharp > final_sharp:

				final_risk = risks[i]
				final_return = returns[i]
				final_ws     = ws[i]
				final_sharp  = sharp	
			

	return final_risk, final_return, final_ws, final_sharp


#markowitz
def markowitz(funddf, bounds):


        rf = const.rf

	funddfr = funddf.pct_change()

	funddfr = funddfr.dropna()

	
	final_risk = 0
	final_return = 0
	final_ws = []
	final_sharp = -1000
	final_codes = []
	
	
	codes = funddfr.columns

	return_rate = []
	for code in codes:
		return_rate.append(funddfr[code].values)


	#print return_rate	
	risks, returns, ws = fin.efficient_frontier(return_rate, bounds)

	
	for j in range(0, len(risks)):

		sharp = (returns[j] - rf) / risks[j]
		if sharp > final_sharp:

			final_risk = risks[j]
			final_return = returns[j]
			final_ws     = ws[j]
			final_sharp  = sharp	


	return final_risk, final_return, final_ws, final_sharp



def portfolio(indexdf, funddf, fund_rank):
					

	fundweights = {}
	indexrisk, indexreturn ,indexws, indexsharp = indexallocation(indexdf)

	for i in range(0, len(indexws)):
		indexw = indexws[i]
		codes = fund_rank[i]
		fundrisk, fundreturn, fundws, fundsharp = technicallocation(funddf, codes)

		for j in range(0 ,len(fundws)):

			code = codes[j]
			w    = fundws[j]

			weight = fundweights.setdefault(code, 0)						
			weight = weight + w * indexw

			fundweights[code] = weight


	ws = {}

	for k,v in fundweights.items():
		if v < 0.01:
			continue
		ws[k] = v			


	sum = 0
	for k, v in ws.items():
		sum = sum + v


	for k in ws.keys():
		ws[k] = ws[k] / sum				


	return ws
	

def portfolio_value(funddf, ws):

	funddf = funddf.dropna()

	codes = ws.keys()
	
	values = []
	w      = []
	for code in codes:
		vs = funddf[code].values
		tran_vs = []
		for v in vs:
			tran_vs.append(v / vs[0])
		
		values.append(tran_vs)
		w.append(ws[code])



	pvs = []
	num = len(values[0])
	for i in range(0, num):
        	v = 0
        	for j in range(0, len(ws)):
                	v = v + values[j][i] * w[j]
        	pvs.append(v)
	
	prs = []
	for i in range(1, len(pvs)):
		prs.append(pvs[i] / pvs[i-1] - 1)		

	return pvs, prs						
	

#利用blacklitterman做战略资产配置		
def strategicallocation(delta,	weq, V, tau, P, Q):

	P = np.array(P)
	Q = np.array(Q)

	tauV = tau * V

	Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])	
	
	res = fin.black_litterman(delta, weq, V, tau, P, Q, Omega)	
	
	return re


#资产配置
def asset_allocation(start_date, end_date, fund_tags, P, Q):

	
	delta = 2.5
	tau = 0.05

	ps = []
	for p in P:
		ps.append(np.array(p))

	P = np.array(ps)	

	qs = []
	for q in Q:
		qs.append(np.array(q))

	Q = np.array(qs)


	indexdf = data.index_value(start_date, end_date, [const.largecap_code, const.smallcap_code])

	indexdfr = indexdf.pct_change().dropna()


	indexrs = []
	for code in indexdfr.columns:
		indexrs.append(indexdfr[code].values)

	#print indexdfr

	sigma = np.cov(indexrs)

	#print type(sigma)
	#print sigma
	#print np.cov(indexrs)
	#print indexdfr	


	weq = np.array([0.5, 0.5])
	tauV = tau * sigma	
	Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])
	er, ws, lmbda = fin.black_litterman(delta, weq, sigma, tau, P, Q, Omega)


	sum = 0
	for w in ws:
		sum = sum + w
	for i in range(0, len(ws)):
		ws[i] = 1.0 * ws[i] / sum		


	#print er
	indexws = ws


	largecap          =   fund_tags['largecap']
	smallcap          =   fund_tags['smallcap']
	risefitness       =   fund_tags['risefitness']			
	declinefitness    =   fund_tags['declinefitness']			
	oscillafitness    =   fund_tags['risefitness']			
	growthfitness     =   fund_tags['risefitness']			
	valuefitness      =   fund_tags['risefitness']			



	print largecap	
	print smallcap
	#print risefitness
	#print declinefitness
	#print oscillafitness
	#print growthfitness
	#print valuefitness
	print 


	code_set = set()
	tag_set  = set()


	#for code in largecap:
		
	
	return 0
