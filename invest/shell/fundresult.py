#coding=utf8


import sys
import string
from datetime import datetime
import numpy as np
from numpy import *
import pandas as pd


#start_date = '2010-01-08'
#end_date   = '2010-12-31'

#start_date = '2011-01-07'
#end_date   = '2011-12-30'

#start_date = '2012-01-06'
#end_date   = '2012-12-28'


#start_date = '2013-01-04'
#end_date   = '2013-12-27'

#start_date = '2014-01-03'
#end_date   = '2014-12-31'


start_date = '2015-01-09'
end_date   = '2015-12-31'


lines = open('./data/testfunds','r').readlines()
fundvs = {}
for line in lines:
        vec = line.split()
        code = string.atoi(vec[0].strip())
        date = datetime.strptime(vec[1].strip(),'%Y-%m-%d')
        if (date < datetime.strptime(start_date, '%Y-%m-%d')) or (date > datetime.strptime(end_date,'%Y-%m-%d')):
                continue
    #print date
        v    = string.atof(vec[2].strip())
        vs   = fundvs.setdefault(code, [])
        vs.append(v)


lines = open('./data/weights','r')
codes = []
ws = []    


for line in lines:
    vec = line.strip().split()
    codes.append(string.atoi(vec[0]))
    ws.append(string.atof(vec[1]))


values = []
for code in codes:
    values.append(fundvs[code])


pvs = []
num = len(values[0])
for i in range(0, num):
        v = 0
        for j in range(0, len(ws)):
                v = v + values[j][i] * ws[j]
        pvs.append(v)

base = pvs[0]
for  i in range(0, len(pvs)):
    pvs[i] = pvs[i] / base    


prs = []
for i in range(1, len(pvs)):
    prs.append(pvs[i] / pvs[i-1] - 1)


rf = 0.02
sharp = (np.mean(prs) * 52 - rf) / (np.std(prs) * (52 ** 0.5))
inv_list =  array(pvs)
running_max = pd.expanding_max(inv_list)
diff = (inv_list - running_max)/running_max
print 'sharp : ', sharp, 'returns :',np.mean(prs) * 52, 'risk:',np.std(prs), 'max drawdown : ', min(0, diff.min())



allsharps = []
allreturns = []
allrisks = []

for k , v in fundvs.items():

    rs = []
    for i in range(1, len(v)):
        rs.append(v[i] / v[i-1] - 1)

    allsharps.append((np.mean(rs) * 52 - rf) / (np.std(rs) *  (52 ** 0.5)))
    #print k, np.mean(rs)
    allreturns.append(np.mean(rs))
    allrisks.append(np.std(rs))


allsharps.sort()
allreturns.sort()
allrisks.sort()

print 'middle sharp : ' ,allsharps[len(allsharps) / 2]
print 'middle returns: ',allreturns[len(allreturns) / 2] * 52
print 'middle risks: ',allrisks[len(allrisks) / 2]



#print pvs

#test['diff'] = diff
#test['diff'].plot(grid=True, figsize=(16,10))
