#coding=utf8


import sys
import string
from datetime import datetime
import numpy as np
from numpy import *
import pandas as pd


start_date = datetime.strptime('2015-01-04','%Y-%m-%d')
end_date   = datetime.strptime('2015-12-27','%Y-%m-%d')


lines = open(sys.argv[1], 'r')
funds = {}
for line in lines:
        vec = line.split()
        code = string.atoi(vec[0])
	d = datetime.strptime(vec[1].strip(), '%Y-%m-%d')
	if d < start_date:
		continue
	if d > end_date:
		continue
        r    = string.atof(vec[2])
        rs   = funds.setdefault(code, [])
	rs.append(r)



lines = open(sys.argv[2],'r')
codes = []
ws = []	


for line in lines:
	vec = line.strip().split()
	codes.append(string.atoi(vec[0]))
	ws.append(string.atof(vec[1]))


values = []
for code in codes:
	values.append(funds[code])


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
print 'sharp : ', sharp, 'returns :',np.mean(prs) * 52, 'risk:',np.std(prs)	



allsharps = []
allreturns = []
allrisks = []
for k , v in funds.items():

	rs = []
	for i in range(1, len(v)):
		rs.append(v[i] / v[i-1] - 1)

	allsharps.append((np.mean(rs) * 52 - rf) / (np.std(rs) *  (52 ** 0.5)))
	allreturns.append(np.mean(rs))
	allrisks.append(np.std(rs))


allsharps.sort()
allreturns.sort()
allrisks.sort()
print 'middle sharp : ' ,allsharps[len(allsharps) / 2]
print 'middle returns: ',allreturns[len(allreturns) / 2] * 52
print 'middle risks: ',allrisks[len(allrisks) / 2]


inv_list =  array(pvs)
running_max = pd.expanding_max(inv_list)
diff = (inv_list - running_max)/running_max

print 'max drawdown : ',min(0, diff.min())
#print pvs

#test['diff'] = diff
#test['diff'].plot(grid=True, figsize=(16,10))
