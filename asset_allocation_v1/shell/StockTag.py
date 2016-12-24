#coding=utf8


import os
import sys
import math
sys.path.append("windshell")
import Data
import Const
import string
from numpy import *
import numpy as np
import pandas as pd
import Financial as fin
import FundIndicator as fi
import AllocationData

from Const import datapath
from dateutil.parser import parse

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
        fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - Const.rf) / np.std(rs))


    x = fitness
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
    sorted_fitness = sorted_x


    result = []
    for i in range(0, (int)(math.ceil(len(sorted_fitness) * ratio))):
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
        fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - Const.rf) / np.std(rs))


    x = fitness
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
    sorted_fitness = sorted_x


    result = []
    for i in range(0, (int)(math.ceil(len(sorted_fitness) * ratio))):
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
        fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - Const.rf) / np.std(rs))


    x = fitness
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
    sorted_fitness = sorted_x


    result = []
    for i in range(0, (int)(math.ceil(len(sorted_fitness) * ratio))):
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
        fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - Const.rf) / np.std(rs))


    x = fitness
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
    sorted_fitness = sorted_x

    result = []
    for i in range(0, (int)(math.ceil(len(sorted_fitness) * ratio))):
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
        fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - Const.rf) / np.std(rs))


    x = fitness
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
    sorted_fitness = sorted_x


    result = []
    for i in range(0, (int)(math.ceil(len(sorted_fitness) * ratio))):
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
        fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - Const.rf) / np.std(rs))


    x = fitness
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
    sorted_fitness = sorted_x


    result = []
    for i in range(0, (int)(math.ceil(len(sorted_fitness) * ratio))):
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
        fitness[code] = (np.mean(rs), np.std(rs), (np.mean(rs) - Const.rf) / np.std(rs))


    x = fitness
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1][2], reverse=True)
    sorted_fitness = sorted_x


    result = []
    for i in range(0, (int)(math.ceil(len(sorted_fitness) * ratio))):
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
    for i in range(0, (int)(math.ceil(len(sorted_position) * ratio))):
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
    for i in range(0, (int)(math.ceil(len(sorted_largecapprefer) * ratio))):
        # if (sorted_largecapprefer[i][1]) > 0.92:
        #     result.append(sorted_largecapprefer[i])
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
    for i in range(0, (int)(math.ceil(len(sorted_smallcapprefer) * ratio))):
        # if (sorted_smallcapprefer[i][1]) > 0.95:
        #     result.append(sorted_smallcapprefer[i])
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
    for i in range(0, (int)(math.ceil(len(sorted_growthcapprefer) * ratio))):
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
    for i in range(0, (int)(math.ceil(len(sorted_valuecapprefer) * ratio))):
        result.append(sorted_valuecapprefer[i])


    return result


def ratebondprefer(funddf, indexdf, ratio):

    ratebondprefer = {}

    funddfr = funddf.pct_change()
    indexdfr = indexdf.pct_change()

    #cols   = indexdfr.columns

    indexr = indexdfr.values

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
        #print len(p)
        #print len(m)
        #print type(p)
        #print type(m)
        p = np.array(p).reshape(1, len(p))
        m = np.array(m).reshape(1, len(m))
        ratebondprefer[col] = np.corrcoef(p, m)[0][1]


    x = ratebondprefer
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
    sorted_ratebondprefer = sorted_x


    result = []
    for i in range(0, (int)(math.ceil(len(sorted_ratebondprefer) * ratio))):
        result.append(sorted_ratebondprefer[i])

    return result


def creditbondprefer(funddf, indexdf, ratio):

    creditbondprefer = {}

    funddfr = funddf.pct_change()
    indexdfr = indexdf.pct_change()

    #cols   = indexdfr.columns

    indexr = indexdfr.values

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
        p = np.array(p).reshape(1, len(p))
        m = np.array(m).reshape(1, len(m))
        creditbondprefer[col] = np.corrcoef(p, m)[0][1]


    x = creditbondprefer
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
    sorted_creditbondprefer = sorted_x

    result = []
    for i in range(0, (int)(math.ceil(len(sorted_creditbondprefer) * ratio))):
        result.append(sorted_creditbondprefer[i])

    return result


def convertiblebondprefer(funddf, indexdf, ratio):

    convertiblebondprefer = {}

    funddfr = funddf.pct_change()
    indexdfr = indexdf.pct_change()

    #cols   = indexdfr.columns

    indexr = indexdfr.values

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
        p = np.array(p).reshape(1, len(p))
        m = np.array(m).reshape(1, len(m))
        convertiblebondprefer[col] = np.corrcoef(p, m)[0][1]

    x = convertiblebondprefer
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
    sorted_convertiblebondprefer = sorted_x

    result = []
    for i in range(0, (int)(math.ceil(len(sorted_convertiblebondprefer) * ratio))):
        result.append(sorted_convertiblebondprefer[i])

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
                m.append(indexrs[i][0])

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
                m.append(indexrs[i][0])

        #print p
        hm[col] = fin.hm(p, m, rf)

    return hm


#输出每一个基金的标签
def tagstockfund(allocationdata, funddf, indexdf):

    dates = indexdf.index.values
    
    dates.sort()
    end_date   = parse(str(dates[-1])).strftime('%Y-%m-%d')
    start_date = parse(str(dates[0])).strftime('%Y-%m-%d')


    capindexdf = indexdf[['399314.SZ', '399316.SZ']]
    largecapindexdf = indexdf[['399314.SZ']]
    smallcapindexdf = indexdf[['399316.SZ']]
    hs300indexdf = indexdf[['000300.SH']]
    growthvalueindexdf = indexdf[['399372.SZ', '399373.SZ', '399376.SZ', '399377.SZ']]


    codes = funddf.columns

    positiondf = Data.fund_position(start_date, end_date)
    columns = set(positiondf.columns)
    tmp_codes = []
    for code in codes:
        if code in columns:
            tmp_codes.append(code)
    codes = tmp_codes


    positiondf = positiondf[codes]


    largecapfitness_result    = largecapfitness(funddf, capindexdf, 0.5)
    smallcapfitness_result    = smallcapfitness(funddf, capindexdf, 0.5)
    risefitness_result    = risefitness(funddf, hs300indexdf, 0.5)
    declinefitness_result     = declinefitness(funddf, hs300indexdf, 0.5)
    oscillationfitness_result = oscillationfitness(funddf, hs300indexdf,  0.5)
    growthfitness_result      = growthfitness(funddf, growthvalueindexdf, 0.5)
    valuefitness_result       = valuefitness(funddf,  growthvalueindexdf, 0.5)
    positionprefer_result     = positionprefer(positiondf, 0.5)
    largecapprefer_result     = largecapprefer(funddf, largecapindexdf, 0.5)
    smallcapprefer_result     = smallcapprefer(funddf, smallcapindexdf, 0.5)
    growthcapprefer_result    = growthcapprefer(funddf, growthvalueindexdf, 0.5)
    valuecapprefer_result     = valuecapprefer(funddf, growthvalueindexdf, 0.5)


    #print 'largecap'
    largecapfitness_set =  set()
    for k,v in largecapfitness_result:
        largecapfitness_set.add(k)
        #print k, v


    #print
    #print 'smallcap'
    smallcapfitness_set = set()
    for k,v in smallcapfitness_result:
        smallcapfitness_set.add(k)
        #print k, v


    #print
    #print 'rise'
    risefitness_set = set()
    for k,v in risefitness_result:
        risefitness_set.add(k)
        #print k, v


    #print
    declinefitness_set = set()
    #print 'decline'
    for k,v in declinefitness_result:
        declinefitness_set.add(k)
        #print k, v


    #print 'oscillation'
    oscillation_set = set()
    for k,v in oscillationfitness_result:
        oscillation_set.add(k)
        #print k, v
    #print


    #print 'growth'
    growthfitness_set = set()
    for k,v in growthfitness_result:
        growthfitness_set.add(k)
        #print k, v

    #print

    #print 'value'
    valuefitness_set = set()
    for k,v in valuefitness_result:
        valuefitness_set.add(k)
        #print k, v


    #print
    #print 'positionprefer'
    positionprefer_set = set()
    for k,v in positionprefer_result:
        positionprefer_set.add(k)
        #print k, v


    #print
    #print 'largecapprefer'
    largecapprefer_set = set()
    for k, v in largecapprefer_result:
        largecapprefer_set.add(k)
        #print k, v

    #print
    #print 'smallcapprefer'
    smallcapprefer_set = set()
    for k, v in smallcapprefer_result:
        smallcapprefer_set.add(k)
        #print k, v
    #print largecapfitness


    #print
    #print 'grwothcapprefer'
    growthcapprefer_set = set()
    for k, v in growthcapprefer_result:
        growthcapprefer_set.add(k)
        #print k, v


    #print
    #print 'valuecapprefer'
    valuecapprefer_set = set()
    for k, v in valuecapprefer_result:
        valuecapprefer_set.add(k)
        #print k, v



    final_codes = set()
    fund_tags = {}

    #print 'rise'
    codes = []
    for code in positionprefer_set:
        if code in risefitness_set:
            #print code
            final_codes.add(code)
            codes.append(code)
    fund_tags['risefitness'] = codes        

    #print 'decline'
    codes = []
    for code in declinefitness_set:
        if code not in positionprefer_set:
            #print code
            final_codes.add(code)
            codes.append(code)
    fund_tags['declinefitness'] = codes        

    #print 'oscillation'
    codes = []
    for code in oscillation_set:
        final_codes.add(code)
        codes.append(code)
    fund_tags['oscillationfitness'] = codes        

    #print 'largecap'
    codes = []
    for code in largecapprefer_set:
        if code in largecapfitness_set:
            #print code
            final_codes.add(code)
            codes.append(code)
    fund_tags['largecap'] = codes



    #print 'smallcap'
    codes = []    
    for code in smallcapprefer_set:
        if code in smallcapfitness_set:
            #print code
            final_codes.add(code)
            codes.append(code)
    fund_tags['smallcap'] = codes



    #print  'growth'
    codes = []
    for code in growthcapprefer_set:
        if code in growthfitness_set:
            #print code
            final_codes.add(code)
            codes.append(code)
    fund_tags['growthfitness'] = codes


    #print 'value'
    codes = []
    for code in valuecapprefer_set:
        if code in valuefitness_set:
            #print code
            final_codes.add(code)
            codes.append(code)
    fund_tags['valuefitness'] = codes

    #print len(final_codes)
    #print final_codes


    funddf = funddf[list(final_codes)]
    #print 'tm'
    #print tmmeasure(funddf, hs300indexdf)


    #print
    #print 'hm'
    #print hmmeasure(funddf, hs300indexdf)

    codes = list(final_codes)
    funddf = funddf[codes]
    #funddf = data.fund_value(start_date, end_date)
    #funddf = funddf[codes]

    #funds = set()

    indicator_datas = []
    indicator_codes = []
    tag_columns = ['high_position_prefer','low_position_prefer','largecap_prefer','smallcap_prefer','growth_prefer','value_prefer','largecap_fitness','smallcap_fitness','rise_fitness','decline_fitness','oscillation_fitness','growth_fitness','value_fitness']

    for code in final_codes:

        indicator_codes.append(code)
        labels = []

        if code in positionprefer_set:
            labels.append(1)
            labels.append(0)
        else:
            labels.append(0)
            labels.append(1)

        if code in largecapprefer_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in smallcapprefer_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in growthcapprefer_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in valuecapprefer_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in largecapfitness_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in smallcapfitness_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in risefitness_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in declinefitness_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in oscillation_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in growthfitness_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in valuefitness_set:
            labels.append(1)
        else:
            labels.append(0)

        indicator_datas.append(labels)

    indicator_df = pd.DataFrame(indicator_datas, index = indicator_codes, columns=tag_columns)
    indicator_df.index.name = 'code'
    indicator_df.to_csv(datapath('stock_label_' + end_date + '.csv'))

    allocationdata.stock_fund_label[end_date] = indicator_df

    return list(final_codes) , fund_tags

def tag_stock_fund_new(day, df_nav_fund, df_nav_index):
    daystr = day.strftime("%Y-%m-%d")
    
    dates = df_nav_index.index.values
    dates.sort()
    end_date   = parse(str(dates[-1])).strftime('%Y-%m-%d')
    start_date = parse(str(dates[0])).strftime('%Y-%m-%d')


    capindexdf = df_nav_index[['399314.SZ', '399316.SZ']]
    largecapindexdf = df_nav_index[['399314.SZ']]
    smallcapindexdf = df_nav_index[['399316.SZ']]
    hs300indexdf = df_nav_index[['000300.SH']]
    growthvalueindexdf = df_nav_index[['399372.SZ', '399373.SZ', '399376.SZ', '399377.SZ']]


    codes = df_nav_fund.columns

    positiondf = Data.fund_position(start_date, end_date)
    columns = set(positiondf.columns)
    tmp_codes = []
    for code in codes:
        if code in columns:
            tmp_codes.append(code)
    codes = tmp_codes


    positiondf = positiondf[codes]


    largecapfitness_result    = largecapfitness(df_nav_fund, capindexdf, 0.5)
    smallcapfitness_result    = smallcapfitness(df_nav_fund, capindexdf, 0.5)
    risefitness_result    = risefitness(df_nav_fund, hs300indexdf, 0.5)
    declinefitness_result     = declinefitness(df_nav_fund, hs300indexdf, 0.5)
    oscillationfitness_result = oscillationfitness(df_nav_fund, hs300indexdf,  0.5)
    growthfitness_result      = growthfitness(df_nav_fund, growthvalueindexdf, 0.5)
    valuefitness_result       = valuefitness(df_nav_fund,  growthvalueindexdf, 0.5)
    positionprefer_result     = positionprefer(positiondf, 0.5)
    largecapprefer_result     = largecapprefer(df_nav_fund, largecapindexdf, 0.5)
    smallcapprefer_result     = smallcapprefer(df_nav_fund, smallcapindexdf, 0.5)
    growthcapprefer_result    = growthcapprefer(df_nav_fund, growthvalueindexdf, 0.5)
    valuecapprefer_result     = valuecapprefer(df_nav_fund, growthvalueindexdf, 0.5)

    data = {
        'high_position_prefer': {k:1 for (k, v) in positionprefer_result},
        'largecap_prefer': {k:1 for (k, v) in largecapprefer_result},
        'smallcap_prefer': {k:1 for (k, v) in smallcapprefer_result},
        'growth_prefer': {k:1 for (k, v) in growthcapprefer_result},
        'value_prefer': {k:1 for (k, v) in valuecapprefer_result},
        'largecap_fitness': {k:1 for (k, v) in largecapfitness_result},
        'smallcap_fitness': {k:1 for (k, v) in smallcapfitness_result},
        'rise_fitness': {k:1 for (k, v) in risefitness_result},
        'decline_fitness': {k:1 for (k, v) in declinefitness_result},
        'oscillation_fitness': {k:1 for (k, v) in oscillationfitness_result},
        'growth_fitness': {k:1 for (k, v) in growthfitness_result},
        'value_fitness': {k:1 for (k, v) in valuefitness_result}
    };

    df_label = pd.DataFrame(data, columns=["high_position_prefer","largecap_prefer","smallcap_prefer","growth_prefer","value_prefer","largecap_fitness","smallcap_fitness","rise_fitness","decline_fitness","oscillation_fitness","growth_fitness","value_fitness"])
    df_label.index.name = 'code'
    df_label.fillna(0, inplace=True)
    df_label = df_label.applymap(lambda x: int(round(x)))
    df_label.to_csv(datapath('stock_blabel_' + daystr + '.csv'))

    columns = ['largecap', 'smallcap', 'rise', 'decline', 'oscillation', 'growth', 'value']
    df_result = pd.DataFrame(0, index=df_label.index, columns=columns)

    mask = (df_label['largecap_prefer'] == 1) & (df_label['largecap_fitness'] == 1) 
    df_result.loc[mask, 'largecap'] = 1

    mask = (df_label['smallcap_prefer'] == 1) & (df_label['smallcap_fitness'] == 1) 
    df_result.loc[mask, 'smallcap'] = 1
    
    mask = (df_label['high_position_prefer'] == 1) & (df_label['rise_fitness'] == 1) 
    df_result.loc[mask, 'rise'] = 1
    
    mask = (df_label['high_position_prefer'] == 0) & (df_label['decline_fitness'] == 1) 
    df_result.loc[mask, 'decline'] = 1
    
    mask = (df_label['oscillation_fitness'] == 1)
    df_result.loc[mask, 'oscillation'] = 1
    
    mask = (df_label['growth_prefer'] == 1) & (df_label['growth_fitness'] == 1) 
    df_result.loc[mask, 'growth'] = 1

    mask = (df_label['value_prefer'] == 1) & (df_label['value_fitness'] == 1) 
    df_result.loc[mask, 'value'] = 1
    
    df_result.to_csv(datapath('stock_label_' + daystr + '.csv'))

    return df_result


def tagbondfund(allocationdata, funddf, indexdf):


    dates = indexdf.index.values
    
    dates.sort()
    end_date   = parse(str(dates[-1])).strftime('%Y-%m-%d')
    start_date = parse(str(dates[0])).strftime('%Y-%m-%d')


    #funddf = Data.bond_value(start_date, end_date)
    #funddf = funddf[codes]

    csibondindexdf              = indexdf[[Const.csibondindex_code]]
    ratebondindexdf             = indexdf[[Const.ratebondindex_code]]

    creditbondindexdf           = indexdf[[Const.credictbondindex_code]]
    convertiblebondindexdf      = indexdf[[Const.convertiblebondindex_code]]

    ratebondprefer_result       =  ratebondprefer(funddf, ratebondindexdf,0.5)
    creditbondprefer_result     =  creditbondprefer(funddf, creditbondindexdf, 0.5)
    convertiblebondprefer_result=  convertiblebondprefer(funddf, convertiblebondindexdf, 0.5)

    #print ratebondprefer_result
    #print creditbondprefer_result
    #print convertiblebondprefer_result

    #print 'ratebondprefer'
    ratebondprefer_set = set()
    for k,v in ratebondprefer_result:
        ratebondprefer_set.add(k)

    #print 'ratebondprefer'
    creditbondprefer_set = set()
    for k,v in creditbondprefer_result:
        creditbondprefer_set.add(k)


    convertiblebondprefer_set = set()
    for k,v in convertiblebondprefer_result:
        convertiblebondprefer_set.add(k)

    #print creditbondprefer_set
    #print convertiblebondprefer_set

    final_codes = set()

    for code in ratebondprefer_set:
        final_codes.add(code)

    for code in creditbondprefer_set:
        final_codes.add(code)

    for code in convertiblebondprefer_set:
        final_codes.add(code)

    fund_tags = {}

    codes = []
    for code in ratebondprefer_set:
        if code in final_codes:
            codes.append(code)
    fund_tags['ratebond'] = codes

    codes = []
    for code in creditbondprefer_set:
        if code in final_codes:
            codes.append(code)
    fund_tags['creditbond'] = codes

    codes = []
    for code in convertiblebondprefer_set:
        if code in final_codes:
            codes.append(code)
    fund_tags['convertiblebond'] = codes


    indicator_datas = []
    indicator_codes = []
    tag_columns = ['ratebond','creditbond','convertiblebond']


    for code in final_codes:

        indicator_codes.append(code)
        labels = []

        if code in ratebondprefer_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in creditbondprefer_set:
            labels.append(1)
        else:
            labels.append(0)

        if code in convertiblebondprefer_set:
            labels.append(1)
        else:
            labels.append(0)

        indicator_datas.append(labels)


    indicator_df = pd.DataFrame(indicator_datas, index = indicator_codes, columns=tag_columns)
    indicator_df.index.name = 'code'
    indicator_df.to_csv(datapath('bond_label_' + end_date + '.csv'))

    allocationdata.bond_fund_label[end_date] = indicator_df

    return list(final_codes), fund_tags

def tag_bond_fund_new(day, df_nav_fund, df_nav_index):
    daystr = day.strftime("%Y-%m-%d")

    # csibondindexdf              = indexdf[[Const.csibondindex_code]]

    ratebondindexdf             = df_nav_index[[Const.ratebondindex_code]]
    creditbondindexdf           = df_nav_index[[Const.credictbondindex_code]]
    convertiblebondindexdf      = df_nav_index[[Const.convertiblebondindex_code]]

    ratebondprefer_result       =  ratebondprefer(df_nav_fund, ratebondindexdf,0.5)
    creditbondprefer_result     =  creditbondprefer(df_nav_fund, creditbondindexdf, 0.5)
    convertiblebondprefer_result=  convertiblebondprefer(df_nav_fund, convertiblebondindexdf, 0.5)

    data = {
        'ratebond': {k:1 for (k, v) in ratebondprefer_result},
        'creditbond': {k:1 for (k, v) in creditbondprefer_result},
        'convertiblebond': {k:1 for (k, v) in convertiblebondprefer_result},
    };

    columns=["ratebond","creditbond", "convertiblebond"]
    
    df_label = pd.DataFrame(data, columns=columns)
    df_label.index.name = 'code'
    df_label.to_csv(datapath('bond_blabel_' + daystr + '.csv'))
    df_label.fillna(0, inplace=True)
    df_label = df_label.applymap(lambda x: int(round(x)))
    df_label.to_csv(datapath('bond_label_' + daystr + '.csv'))

    return df_label



if __name__ == '__main__':


    train_start = '2010-01-01'
    train_end   = '2012-12-31'

    codes       =   main.fundfilter(train_start, train_end)
    #print codes

    fund_codes  =   tagfunds(train_start, train_end, codes)

    print fund_codes

    #risk, returns, ws, sharp = pf.markowitz()


    '''
    funddf = data.fund_value(train_start, train_end)
    funddf = funddf[codes]


    capindexdf     = data.index_value(train_start, train_end, ['399314.SZ', '399316.SZ'])
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
    largecapprefer_set = set()
    for k, v in largecapprefer:
        largecapprefer_set.add(k)
        print k, v

    print
    print 'smallcapprefer'
    smallcapprefer_set = set()
    for k, v in smallcapprefer:
        smallcapprefer_set.add(k)
        print k, v
    #print largecapfitness


    print
    print 'grwothcapprefer'
    growthcapprefer_set = set()
    for k, v in growthcapprefer:
        growthcapprefer_set.add(k)
        print k, v
    print
    print 'valuecapprefer'
    valuecapprefer_set = set()
    for k, v in valuecapprefer:
        valuecapprefer_set.add(k)
        print k, v



    final_codes = set()
    print
    print 'rise'
    for code in positionprefer_set:
        if code in risefitness_set:
            print code
            final_codes.add(code)


    print
    print 'largecap'
    for code in largecapprefer_set:
        if code in largecapfitness_set:
            print code
            final_codes.add(code)


    print
    print 'smallcap'
    for code in smallcapprefer_set:
        if code in smallcapfitness_set:
            print code
            final_codes.add(code)


    print
    print  'growth'
    for code in growthcapprefer_set:
        if code in growthfitness_set:
            print code
            final_codes.add(code)


    print
    print 'value'
    for code in valuecapprefer_set:
        if code in valuefitness_set:
            print code
            final_codes.add(code)


    print
    #print len(final_codes)
    print final_codes



    funddf = funddf[list(final_codes)]
    print
    #print 'tm'
    #print tmmeasure(funddf, hs300indexdf)


    print
    #print 'hm'
    #print hmmeasure(funddf, hs300indexdf)


    codes = list(final_codes)
    funddf = data.fund_value(train_start, train_end)
    #funddf = funddf[codes]


    funds = set()

    print 'large'
    codes = []
    for code in largecapfitness_set:
        if code in final_codes and (not code in funds):
            codes.append(code)

    largecapfitness_df = funddf[codes]
    sharps = fi.fund_sharp(largecapfitness_df)
    print sharps
    funds.add(sharps[0][0])


    print 'small'
    codes = []
    for code in smallcapfitness_set:
        if code in final_codes and (not code in funds):
            codes.append(code)

    smallcapfitness_df = funddf[codes]
    sharps = fi.fund_sharp(smallcapfitness_df)
    print sharps
    funds.add(sharps[0][0])


    print 'rise'
    codes = []
    for code in risefitness_set:
        if code in final_codes and (not code in funds):
            codes.append(code)

    risefitness_df = funddf[codes]
    sharps = fi.fund_sharp(risefitness_df)
    print sharps
    funds.add(sharps[0][0])


    print 'decline'
    codes = []
    for code in declinefitness_set:
        if code in final_codes and (not code in funds):
            codes.append(code)

    declinefitness_df = funddf[codes]
    sharps = fi.fund_sharp(declinefitness_df)
    print sharps
    funds.add(sharps[0][0])


    print 'oscillation'
    codes = []
    for code in oscillation_set:
        if code in final_codes and (not code in funds):
            codes.append(code)

    oscillationfitness_df = funddf[codes]
    sharps = fi.fund_sharp(oscillationfitness_df)
    print sharps
    funds.add(sharps[0][0])


    print 'growth'
    codes = []
    for code in growthfitness_set:
        if code in final_codes and (not code in funds):
            codes.append(code)

    growthfitness_df = funddf[codes]
    sharps = fi.fund_sharp(growthfitness_df)
    print sharps
    funds.add(sharps[0][0])


    print 'value'
    codes = []
    for code in valuefitness_set:
        if code in final_codes and (not code in funds):
            codes.append(code)

    valuefitness_df = funddf[codes]
    sharps = fi.fund_sharp(valuefitness_df)
    print sharps
    funds.add(sharps[0][0])


    print funds

    '''


