#coding=utf8

import string
import sys
sys.path.append('shell')
import Financial as fin
import numpy as np
import pandas as pd
from datetime import datetime


#start_date = '2007-01-05'
#end_date   = '2009-12-31'

#start_date = '2008-01-04'
#end_date   = '2010-12-31'


#start_date = '2009-01-09'
#end_date   = '2011-12-30'


#start_date = '2010-01-08'
#end_date   = '2012-12-28'


#start_date = '2011-01-07'
#end_date   = '2013-12-27'


start_date = '2012-01-06'
end_date   = '2014-12-31'

#########################################中类资产配置#################################
df = pd.read_csv('./data/index_weekly', index_col='date', parse_dates=[0])
df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]

return_rate = []
dapan = df['dapan'].values
xiaopan = df['xiaopan'].values


dapanrs = []
xiaopanrs = []
for i in range(1, len(dapan)):
	dapanrs.append(dapan[i] / dapan[i-1] - 1)
for i in range(1, len(xiaopan)):
	xiaopanrs.append(xiaopan[i] / xiaopan[i-1] - 1)

return_rate.append(dapanrs)
return_rate.append(xiaopanrs)
	
risks, returns, portfolios = fin.efficient_frontier(return_rate)


rf = 0.025 / 52
n = 0
sharp = (returns[0] - rf) / risks[0]
for i in range(0, len(returns)):
	s = (returns[i] - rf) / risks[i]
	if s > sharp:
		sharp = s
		n     = i

###########################################################################

#print 'sharp : ', sharp, 'return : ', returns[n], 'risk : ', risks[n], 'portfolio : ', portfolios[n]
#print 'annual return : ', returns[n] * 52


dapan_weight = portfolios[n][0]
xiaopan_weight = portfolios[n][1]


up = {}
down = {}
middle = {}
dapan = {}
xiaopan = {}
chengzhang = {}
jiazhi = {}


lines = open('data/fundlabels','r').readlines()
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


#print up
#print middle
#print down
#print dapan
#print xiaopan
#print chengzhang
#print jiazhi

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


#######################################基金数据###########################

f = open('data/codes','r')
lines = f.readlines()
f.close()

codes = set()
for line in lines:
        codes.add(string.atoi(line.strip()))


f = open('./data/dates','r')
line = f.readline()
f.close()
dates = set()
vec = line.strip().split(',')
for d in vec:
        dates.add(d)


lines = open('./data/funds','r').readlines()

fundvs = {}
for line in lines:
        vec = line.split()
        code = string.atoi(vec[0].strip())
        if code not in codes:
                continue
        d    = vec[1].strip()
        date = datetime.strptime(vec[1].strip(),'%Y-%m-%d')
        if (date < datetime.strptime(start_date, '%Y-%m-%d')) or (date >datetime.strptime(end_date,'%Y-%m-%d') or (d not in dates)):
                continue
        v    = string.atof(vec[2].strip())
        vs   = fundvs.setdefault(code, [])
        vs.append(v)

fundrs = {}
for code in fundvs.keys():
        vs = fundvs[code]
        rs = []
        for i in range(1, len(vs)):
                rs.append(vs[i] / vs[i-1] - 1)
        fundrs[code] = rs

#############################################################################################

final_weights = {}

sharp = 0
r = 0
risk = 0
ws = []
#for i in range(2, min(11, len(dapanfund) + 1) ):

for i in range(2, 11):
	frs = []
	for j in range(0, i):
		frs.append(fundrs[dapanfund[j]])

	risks,returns,portfolios = fin.efficient_frontier(frs)						
	for n in range(0, len(risks)):
		s = (returns[n] - rf) / risks[n]
		if s > sharp:
			r = returns[n]
			risk = risks[n]
			sharp = s
			ws = portfolios[n]

#print sharp, r, risk, ws	

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
#for i in range(2, min(11, len(xiaopanfund) + 1)):
for i in range(2, 11):
	frs = []
	for j in range(0, i):
		frs.append(fundrs[xiaopanfund[j]])
	risks,returns,portfolios = fin.efficient_frontier(frs)						
	for n in range(0, len(risks)):
		s = (returns[n] - rf) / risks[n]
		if s > sharp:
			r = returns[n]
			risk = risks[n]
			sharp = s
			ws = portfolios[n]


#print sharp, r, risk, ws	

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


sumw = 0
for w in final_weight:
	sumw = sumw + w

for i in range(0, len(final_weight)):
	final_weight[i] = final_weight[i] / sumw


for i in range(0, len(final_codes)):
	print final_codes[i], final_weight[i]


#print final_weight


fund_values = []
for code in final_codes:
	rs = fundrs[code]	
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


rs = []
for i in range(1 ,len(pvs)):
	rs.append(pvs[i] / pvs[i-1] - 1)

	
sharp = (np.mean(rs) - rf) / np.std(rs)
print sharp, np.mean(rs), np.std(rs)


print final_codes
print final_weight


f = open('./data/weights','w')
for i in range(0, len(final_codes)):
	f.write(str(final_codes[i]) + ' '  + str(final_weight[i]) + '\n')

f.close()

#allsharps = {}
#for k, v in fundrs.items():
#	allsharps[k] = ( (np.mean(v) - rf) / np.std(v))
#allsharps = sorted(allsharps.iteritems(), key=lambda x : x[1], reverse=True)




#for k, v in allsharps:
#	print k, v 
#print '.....'
#print dapan_weight
#print xiaopan_weight 

