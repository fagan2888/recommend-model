#coding=utf8

import string
import sys
sys.path.append('shell')
import Financial as fin
import numpy as np
import pandas as pd


lines = open(sys.argv[1],'r').readlines()

return_rate = []
for line in lines:
	vec = line.strip().split(',')
	rs  = []
	for v in vec:
		rs.append(string.atof(v))
	return_rate.append(rs)

risks, returns, portfolios = fin.efficient_frontier(return_rate)


rf = 0.025 / 52
n = 0
sharp = (returns[0] - rf) / risks[0]
for i in range(0, len(returns)):
	s = (returns[i] - rf) / risks[i]
	if s > sharp:
		sharp = s
		n     = i

print sharp, returns[n], risks[n], portfolios[n]
print returns[n] * 52

print portfolios[n][0]
print portfolios[n][1]


dapan_weight = portfolios[n][0]
xiaopan_weight = portfolios[n][1]


up = {}
down = {}
middle = {}
dapan = {}
xiaopan = {}
chengzhang = {}
jiazhi = {}

lines = open(sys.argv[2],'r').readlines()
for line in lines:
	vec = line.strip().split()
	if 'up' == vec[1].strip():
		up[string.atoi(vec[0])] = string.atof(vec[2])

	if 'down' == vec[1].strip():
		down[string.atoi(vec[0])] = string.atof(vec[2])

	if 'middle' == vec[1].strip():
		middle[string.atoi(vec[0])] = string.atof(vec[2])

	if 'dapan' == vec[1].strip():
		dapan[string.atoi(vec[0])] = string.atof(vec[2])

	if 'xiaopan' == vec[1].strip():
		xiaopan[string.atoi(vec[0])] = string.atof(vec[2])

	if 'chengzhang' == vec[1].strip():
		chengzhang[string.atoi(vec[0])] = string.atof(vec[2])

	if 'jiazhi' == vec[1].strip():
		jiazhi[string.atoi(vec[0])] = string.atof(vec[2])


upfund = []
x = up
up = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
for i in range(0, len(up) / 2):
	upfund.append(up[i][0])
#print upfund



middlefund = []
x = middle
middle = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
for i in range(0, len(middle) / 2):
	middlefund.append(middle[i][0])
#print middlefund


downfund = []
x = down
down = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
for i in range(0, len(down) / 2):
	downfund.append(down[i][0])
#print downfund


dapanfund = []
x = dapan
dapan = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
for i in range(0, len(dapan) / 2):
	dapanfund.append(dapan[i][0])
#print dapanfund


xiaopanfund = []
x = xiaopan
xiaopan = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
for i in range(0, len(xiaopan) / 2):
	xiaopanfund.append(xiaopan[i][0])
#print xiaopanfund


chengzhangfund = []
x = chengzhang
chengzhang = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
for i in range(0, len(chengzhang) / 2):
	chengzhangfund.append(chengzhang[i][0])
#print chengzhangfund


jiazhifund = []
x = jiazhi
jiazhi = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
for i in range(0, len(jiazhi) / 2):
	jiazhifund.append(jiazhi[i][0])
#print jiazhifund


fundreturns = {}
lines = open(sys.argv[3],'r').readlines()
for i in range(1, len(lines)):
	line = lines[i]
	vec = line.strip().split(',')	
	code = string.atoi(vec[0].strip())
	rs = []
	for j in range(1, len(vec) - 1):
		rs.append(string.atof(vec[j]))

	fundreturns[code] = rs



final_weights = {}

sharp = 0
r = 0
risk = 0
ws = []
for i in range(2, 11):
	frs = []
	for j in range(0, i):
		frs.append(fundreturns[dapanfund[j]])
	risks,returns,portfolios = fin.efficient_frontier(frs)						
	for n in range(0, len(risks)):
		s = (returns[n] - rf) / risks[n]
		if s > sharp:
			r = returns[n]
			risk = risks[n]
			sharp = s
			ws = portfolios[n]

print sharp, r, risk, ws	

for i in range(0, len(ws)):
	if ws[i] <= 0.01:
		continue
	code = dapanfund[i]		
	w = final_weights.setdefault(code, 0)
	final_weights[code] = w + ws[i] * dapan_weight	


sharp = 0
r = 0
risk = 0
ws = []
for i in range(2, 11):
	frs = []
	for j in range(0, i):
		frs.append(fundreturns[xiaopanfund[j]])
	risks,returns,portfolios = fin.efficient_frontier(frs)						
	for n in range(0, len(risks)):
		s = (returns[n] - rf) / risks[n]
		if s > sharp:
			r = returns[n]
			risk = risks[n]
			sharp = s
			ws = portfolios[n]


print sharp, r, risk, ws	



for i in range(0, len(ws)):
	if ws[i] <= 0.01:
		continue
	code = xiaopanfund[i]		
	w = final_weights.setdefault(code, 0)
	final_weights[code] = w + ws[i] * xiaopan_weight	



final_codes = []
final_weight = []
for k,v in final_weights.items():
	final_codes.append(k)
	final_weight.append(v)
	print k, v 

sumw = 0
for w in final_weight:
	sumw = sumw + w

for i in range(0, len(final_weight)):
	final_weight[i] = final_weight[i] / sumw

#print final_weight

fund_values = []
for code in final_codes:
	rs = fundreturns[code]	
	vs = []
	v  = 1
	vs.append(v)
	for r in rs:
		v = v * ( 1 + r )
		vs.append(v)
	fund_values.append(vs)


pvs = []
num = len(fund_values[0])
for i in range(0, num):
	v = 0
	for j in range(0, len(final_weight)):
		v = v + fund_values[j][i] * final_weight[j]
	pvs.append(v)

print pvs

rs = []
for i in range(1 ,len(pvs)):
	rs.append(pvs[i] / pvs[i-1] - 1)

	
sharp = (np.mean(rs) - rf) / np.std(rs)
print sharp, np.mean(rs), np.std(rs)


allsharps = {}
for k, v in fundreturns.items():
	allsharps[k] = ( (np.mean(v) - rf) / np.std(v))
allsharps = sorted(allsharps.iteritems(), key=lambda x : x[1], reverse=True)

#for k, v in allsharps:
#	print k, v 
#print '.....'
#print dapan_weight
#print xiaopan_weight 

