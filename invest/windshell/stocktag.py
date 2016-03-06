#coding=utf8


import sys
sys.path.append("windshell")
import main
import data
import const
import string
from numpy import *
import numpy as np
import pandas as pd



#大盘适应度
def largecapfitness(funddf, indexdf, ratio):

	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

        largecaptag = {}

	largecap = []	

	cols   = indexdfr.columns

	indexr = []
	indexr.append(0)
	for i in range(1 ,len(indexdfr[cols[0]].values)):
		indexr.append(indexdfr[cols[0]].values[i] - indexdfr[cols[1]].values[i])
	

	for i in range(4, len(indexr)):
		n = 0
		for j in range(0, 4):
			v = indexr[i - j]
			if v >= 0:
				n = n + 1
			else:
				n = n - 1

		if n == 0:
			largecap.append(0)
		elif n > 0:
			largecap.append(1)
		else:
			largecap.append(-1)		


	for code in funddfr.columns:
	
		fundr = funddfr[code].values				
		largecapr = []
		for i in range(0, len(largecap)):
			tag = largecap[i]
			if tag == 1 and (not isnan(fundr[i + 4])):
				largecapr.append(fundr[i + 4])
	
		largecaptag[code] = largecapr	


	fitness = {}
	for code in largecaptag.keys():
		rs = largecaptag[code]
		fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - const.rf) / np.std(rs)) 	


	x = fitness
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
        sorted_fitness = sorted_x


        result = []
        for i in range(0, (int)(len(sorted_fitness) * ratio)):
                result.append(sorted_fitness[i])


        return result
	


#小盘适应度
def smallcapfitness(funddf, indexdf, ratio):


	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

        smallcaptag = {}

	smallcap = []	

	cols   = indexdfr.columns

	indexr = []
	indexr.append(0)
	for i in range(1 ,len(indexdfr[cols[0]].values)):
		indexr.append(indexdfr[cols[0]].values[i] - indexdfr[cols[1]].values[i])
	

	for i in range(4, len(indexr)):
		n = 0
		for j in range(0, 4):
			v = indexr[i - j]
			if v >= 0:
				n = n + 1
			else:
				n = n - 1

		if n == 0:
			smallcap.append(0)
		elif n > 0:
			smallcap.append(1)
		else:
			smallcap.append(-1)		


	for code in funddfr.columns:
	
		fundr = funddfr[code].values				
		smallcapr = []
		for i in range(0, len(smallcap)):
			tag = smallcap[i]
			if tag == -1 and (not isnan(fundr[i + 4])):
				smallcapr.append(fundr[i + 4])

		smallcaptag[code] = smallcapr	


	fitness = {}
        for code in smallcaptag.keys():
                rs = smallcaptag[code]
                fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - const.rf) / np.std(rs))


        x = fitness
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
        sorted_fitness = sorted_x


        result = []
        for i in range(0, (int)(len(sorted_fitness) * ratio)):
                result.append(sorted_fitness[i])


        return result



#上涨适应度
def risefitness(funddf, indexdf, ratio):

	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

        risetag = {}

	rise = []	

	indexr = indexdfr.values

	for i in range(4, len(indexr)):
		n = 0
		for j in range(0, 4):
			v = indexr[i - j]
			if v >= 0:
				n = n + 1
			else:
				n = n - 1


		if n == 0:
			rise.append(0)
		elif n > 0:
			rise.append(1)
		else:
			rise.append(-1)										

	for code in funddfr.columns:
	
		fundr = funddfr[code].values				
		riser = []
		for i in range(0, len(rise)):
			tag = rise[i]
			if tag == 1 and (not isnan(fundr[i + 4])):
				riser.append(fundr[i + 4])
	
		risetag[code] = riser			

										
	fitness = {}
        for code in risetag.keys():
                rs = risetag[code]
                fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - const.rf) / np.std(rs))


        x = fitness
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
        sorted_fitness = sorted_x


        result = []
        for i in range(0, (int)(len(sorted_fitness) * ratio)):
                result.append(sorted_fitness[i])


        return result



#下跌适应度
def declinefitness(funddf, indexdf, ratio):

	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

        declinetag = {}

	decline = []	

	indexr = indexdfr.values

	for i in range(4, len(indexr)):
		n = 0
		for j in range(0, 4):
			v = indexr[i - j]
			if v >= 0:
				n = n + 1
			else:
				n = n - 1

		if n == 0:
			decline.append(0)
		elif n > 0:
			decline.append(1)
		else:
			decline.append(-1)

	for code in funddfr.columns:
	
		fundr = funddfr[code].values				
		decliner = []
		for i in range(0, len(decline)):
			tag = decline[i]
			if tag == -1 and (not isnan(fundr[i + 4])):
				decliner.append(fundr[i + 4])
	
		declinetag[code] = decliner	
			
							
	fitness = {}
        for code in declinetag.keys():
                rs = declinetag[code]
                fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - const.rf) / np.std(rs))


        x = fitness
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
        sorted_fitness = sorted_x


        result = []
        for i in range(0, (int)(len(sorted_fitness) * ratio)):
                result.append(sorted_fitness[i])


        return result



#震荡适应度
def oscillationfitness(funddf, indexdf, ratio):

	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

        oscillationtag = {}

	oscillation = []	

	indexr = indexdfr.values

	for i in range(4, len(indexr)):
		n = 0
		for j in range(0, 4):
			v = indexr[i - j]
			if v >= 0:
				n = n + 1
			else:
				n = n - 1

		if n == 0:
			oscillation.append(0)
		elif n > 0:
			oscillation.append(1)
		else:
			oscillation.append(-1)		
						
	for code in funddfr.columns:
	
		fundr = funddfr[code].values				
		oscillationr = []
		for i in range(0, len(oscillation)):
			tag = oscillation[i]
			if tag == 0 and not isnan(fundr[i + 4]):
				oscillationr.append(fundr[i + 4])
	
		oscillationtag[code] = oscillationr			

										
	fitness = {}
        for code in oscillationtag.keys():
                rs = oscillationtag[code]
                fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - const.rf) / np.std(rs))


        x = fitness
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
        sorted_fitness = sorted_x


        result = []
        for i in range(0, (int)(len(sorted_fitness) * ratio)):
                result.append(sorted_fitness[i])


        return result


#成长适应度
def growthfitness(funddf, indexdf, ratio):

	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()


        growthcaptag = {}

	growthcap = []	

	cols   = indexdfr.columns

	indexr = []
	indexr.append(0)
	for i in range(1 ,len(indexdfr[cols[0]].values)):
		indexr.append(0.5 * indexdfr['399372.SZ'].values[i]  + 0.5 * indexdfr['399376.SZ'].values[i] - 0.5 * indexdfr['399373.SZ'].values[i] - 0.5 * indexdfr['399377.SZ'].values[i])
	

	for i in range(4, len(indexr)):
		n = 0
		for j in range(0, 4):
			v = indexr[i - j]
			if v >= 0:
				n = n + 1
			else:
				n = n - 1

		if n == 0:
			growthcap.append(0)
		elif n > 0:
			growthcap.append(1)
		else:
			growthcap.append(-1)		


	for code in funddfr.columns:
	
		fundr = funddfr[code].values				
		growthcapr = []
		for i in range(0, len(growthcap)):
			tag = growthcap[i]
			if tag == 1 and (not isnan(fundr[i + 4])):
				growthcapr.append(fundr[i + 4])

		growthcaptag[code] = growthcapr	


	fitness = {}
        for code in growthcaptag.keys():
                rs = growthcaptag[code]
                fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - const.rf) / np.std(rs))


        x = fitness
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
        sorted_fitness = sorted_x


        result = []
        for i in range(0, (int)(len(sorted_fitness) * ratio)):
                result.append(sorted_fitness[i])


        return result



#价值适应度
def valuefitness(funddf, indexdf, ratio):

	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

        valuecaptag = {}

	valuecap = []	

	cols   = indexdfr.columns

	indexr = []
	indexr.append(0)
	for i in range(1 ,len(indexdfr[cols[0]].values)):
		indexr.append(0.5 * indexdfr['399372.SZ'].values[i]  + 0.5 * indexdfr['399376.SZ'].values[i]- 0.5 * indexdfr['399373.SZ'].values[i] - 0.5 * indexdfr['399377.SZ'].values[i])
	

	for i in range(4, len(indexr)):
		n = 0
		for j in range(0, 4):
			v = indexr[i - j]
			if v >= 0:
				n = n + 1
			else:
				n = n - 1

		if n == 0:
			valuecap.append(0)
		elif n > 0:
			valuecap.append(1)
		else:
			valuecap.append(-1)		


	for code in funddfr.columns:
	
		fundr = funddfr[code].values				
		valuecapr = []
		for i in range(0, len(valuecap)):
			tag = valuecap[i]
			if tag == -1 and (not isnan(fundr[i + 4])):
				valuecapr.append(fundr[i + 4])

		valuecaptag[code] = valuecapr	

	fitness = {}
        for code in valuecaptag.keys():
                rs = valuecaptag[code]
                fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - const.rf) / np.std(rs))


        x = fitness
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
        sorted_fitness = sorted_x


        result = []
        for i in range(0, (int)(len(sorted_fitness) * ratio)):
                result.append(sorted_fitness[i])


        return result



#仓位偏好
def positionprefer(funddf, ratio):


	positiontag = {}	
	for col in funddf.columns:
		vs = funddf[col].values
		positiontag[col] = np.mean(vs)


        x = positiontag
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_position = sorted_x


        result = []
        for i in range(0, (int)(len(sorted_position) * ratio)):
                result.append(sorted_position[i])


	return result



#大盘偏好
def largecapprefer(funddf, indexdf, ratio):

	largecapprefer = {}
	

	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

        cols = funddfr.columns
        for col in cols:
                p = []
                m = []
                rs = funddfr[col].values
                #print col, rs
                indexrs = indexdfr.values
                for i in range(0, len(rs)):
                        if isnan(rs[i]) or isnan(indexrs[i]):
                                continue
                        else:
                                p.append(rs[i])
                                m.append(indexrs[i][0])

                #print p
		#print m
		#print np.corrcoef(p,m)
                largecapprefer[col] = np.corrcoef(p, m)[0][1]
		 

        x = largecapprefer
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_largecapprefer = sorted_x

        result = []
        for i in range(0, (int)(len(sorted_largecapprefer) * ratio)):
                result.append(sorted_largecapprefer[i])

        return result
				


def smallcapprefer(funddf, indexdf, ratio):


	smallcapprefer = {}
	

	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

        cols = funddfr.columns
        for col in cols:
                p = []
                m = []
                rs = funddfr[col].values
                #print col, rs
                indexrs = indexdfr.values
                for i in range(0, len(rs)):
                        if isnan(rs[i]) or isnan(indexrs[i]):
                                continue
                        else:
                                p.append(rs[i])
                                m.append(indexrs[i][0])

                #print p
		#print m
		#print np.corrcoef(p,m)
                smallcapprefer[col] = np.corrcoef(p, m)[0][1]


        x = smallcapprefer
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_smallcapprefer = sorted_x


        result = []
        for i in range(0, (int)(len(sorted_smallcapprefer) * ratio)):
                result.append(sorted_smallcapprefer[i])


        return result
				

def growthcapprefer(funddf, indexdf, ratio):

	growthcapprefer = {}
	

	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

	cols = indexdfr.columns

	indexr = []
	indexr.append(0)
	for i in range(1 ,len(indexdfr[cols[0]].values)):
		indexr.append(0.5 * indexdfr['399372.SZ'].values[i]  + 0.5 * indexdfr['399376.SZ'].values[i])

        cols = funddfr.columns
        for col in cols:
                p = []
                m = []
                rs = funddfr[col].values
                #print col, rs
                for i in range(0, len(rs)):
                        if isnan(rs[i]) or isnan(indexr[i]):
                                continue
                        else:
                                p.append(rs[i])
                                m.append(indexr[i])

                #print p
		#print np.corrcoef(p,m)
                growthcapprefer[col] = np.corrcoef(p, m)[0][1]


        x = growthcapprefer
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_growthcapprefer = sorted_x


        result = []
        for i in range(0, (int)(len(sorted_growthcapprefer) * ratio)):
                result.append(sorted_growthcapprefer[i])

        return result
				




def valuecapprefer(funddf, indexdf, ratio):


	valuecapprefer = {}
	
	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

	cols   = indexdfr.columns

	indexr = []
	indexr.append(0)
	for i in range(1 ,len(indexdfr[cols[0]].values)):
		indexr.append(0.5 * indexdfr['399373.SZ'].values[i]  + 0.5 * indexdfr['399377.SZ'].values[i])


        cols = funddfr.columns
        for col in cols:
                p = []
                m = []
                rs = funddfr[col].values
                #print col, rs
                indexrs = indexdfr.values
                for i in range(0, len(rs)):
                        if isnan(rs[i]) or isnan(indexr[i]):
                                continue
                        else:
                                p.append(rs[i])
                                m.append(indexr[i])

                #print p
		#print m
		#print np.corrcoef(p,m)
                valuecapprefer[col] = np.corrcoef(p, m)[0][1]



        x = valuecapprefer
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_valuecapprefer = sorted_x



        result = []
        for i in range(0, (int)(len(sorted_valuecapprefer) * ratio)):
                result.append(sorted_valuecapprefer[i])


        return result
				

#treynor-mazuy测度
def tmmeasure(funddf, indexdf):

	rf = const.rf
	
	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

        tm = {}
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
                tm[col] = fin.tm(p, m, rf)


	return tm								


#henrikson-merton测度
def hmmeasure(funddf, indexdf):


	rf = const.rf
	
	funddfr = funddf.pct_change()
        indexdfr = indexdf.pct_change()

        hm = {}
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
                hm[col] = fin.hm(p, m, rf)

	return hm								



#输出每一个基金的标签
def fundtag(start_date, end_date):
					
	return 0	




if __name__ == '__main__':


	train_start = '2009-01-01'	
	train_end   = '2011-12-31'


	codes = main.fundfilter(train_start, train_end)		
	#print codes


	funddf = data.fund_value(train_start, train_end)
        funddf = funddf[codes]	
	

	
	capindexdf         = data.index_value(train_start, train_end, ['399314.SZ', '399316.SZ'])
	largecapindexdf    = data.index_value(train_start, train_end, ['399314.SZ'])
	smallcapindexdf    = data.index_value(train_start, train_end, ['399316.SZ'])
	hs300indexdf       = data.index_value(train_start, train_end, ['000300.SH'])
	growthvalueindexdf = data.index_value(train_start, train_end, ['399372.SZ','399373.SZ','399376.SZ','399377.SZ'])

	positiondf         = data.fund_position(train_start, train_end)
	positiondf         = positiondf[codes]

        largecapfitness    = largecapfitness(funddf, capindexdf, 0.5)
        smallcapfitness    = smallcapfitness(funddf, capindexdf, 0.5)
	risefitness        = risefitness(funddf, hs300indexdf, 0.5)	
	declinefitness     = declinefitness(funddf, hs300indexdf, 0.5)	
	oscillationfitness = oscillationfitness(funddf, hs300indexdf,  0.5)
	growthfitness      = growthfitness(funddf, growthvalueindexdf, 0.5)
	valuefitness       = valuefitness(funddf,  growthvalueindexdf, 0.5) 
	positionprefer     = positionprefer(positiondf, 0.5)
	largecapprefer     = largecapprefer(funddf, largecapindexdf, 0.5)
	smallcapprefer     = smallcapprefer(funddf, smallcapindexdf, 0.5)
	growthcapprefer    = growthcapprefer(funddf, growthvalueindexdf, 0.5)
	valuecapprefer     = valuecapprefer(funddf, growthvalueindexdf, 0.5)


	print 'largecap'
	largecapfitness_set =  set()
	for k,v in largecapfitness:
		largecapfitness_set.add(k)
		print k, v	

	print 

	print 'smallcap'
	smallcapfitness_set = set()
	for k,v in smallcapfitness:
		smallcapfitness_set.add(k)
		print k, v	

	print 

	print 'rise'
	risefitness_set = set()
	for k,v in risefitness:
		risefitness_set.add(k)		
		print k, v

	print 
	declinefitness_set = set()
	print 'decline'
	for k,v in declinefitness:
		declinefitness_set.add(k)
		print k, v

	print 

	print 'oscillation'
	oscillation_set = set()
	for k,v in oscillationfitness:
		oscillation_set.add(k)
		print k, v

	print 

	print 'growth'
	growthfitness_set = set()	
	for k,v in growthfitness:
		growthfitness_set.add(k)
		print k, v

	print 
	
	print 'value'
	valuefitness_set = set()
	for k,v in valuefitness:
		valuefitness_set.add(k)
		print k, v

	print 
	print 'positionprefer'	
	positionprefer_set = set()
	for k,v in positionprefer:
		positionprefer_set.add(k)
		print k, v


	
	print 
	print 'largecapprefer'	
	largecapperfer_set = set()
	for k, v in largecapprefer:
		largecapprefer_set.add(k)
		print k, v

	print 
	print 'smallcapprefer'	
	for k, v in smallcapprefer:
		print k, v
	#print largecapfitness


	print 
	print 'grwothcapprefer'
	for k, v in growthcapprefer:
		print k, v	


	print 
	print 'valuecapprefer'
	for k, v in valuecapprefer:
		print k, v	



		
