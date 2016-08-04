#coding=utf8


import sys
sys.path.append("shell")
import string
import Financial as fin
import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas as pd


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
    d    = vec[1].strip()
    date = datetime.strptime(vec[1].strip(),'%Y-%m-%d')
    if (date < datetime.strptime(start_date, '%Y-%m-%d')) or (date > datetime.strptime(end_date,'%Y-%m-%d') or (d not in dates)):
        continue
    v    = string.atof(vec[2].strip())
    vs   = fundvs.setdefault(code, [])
    vs.append(v)



#print fundvs

dates = list(dates)
dates.sort()


fundrs = {}
for code in fundvs.keys():
    vs = fundvs[code]
    rs = []
    for i in range(1, len(vs)):
        rs.append(vs[i] / vs[i-1] - 1)
    fundrs[code] = rs
#print fundrs    


df = pd.read_csv('./data/index_weekly', index_col='date', parse_dates=[0])
df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
df = df[ df.index >= datetime.strptime(start_date,'%Y-%m-%d')]
indexvs = df['hs_300'].values


        
indexrs = []
for i in range(1, len(indexvs)):
    indexrs.append(indexvs[i] / indexvs[i-1] - 1)
    


rf = 0.025 / 52
jensen = {}
sortino = {}
for code in fundrs.keys():
    rs = fundrs[code]

    if len(rs) < len(indexrs):
        continue

    jensen[code] = fin.jensen(rs, indexrs, rf)
    sortino[code] = fin.sortino(rs, rf)
    #print jensen[codes[i]]
    #print sortino[codes[i]]

#print jensen
#print sortino
#print dates


x = jensen
sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True) 
sorted_jensen = sorted_x


x = sortino
sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True) 
sorted_sortino = sorted_x



jenson_set = set()
for i in range(0, len(sorted_jensen) / 2):
    k,v = sorted_jensen[i]
    jenson_set.add(k)


sortino_set = set()
for i in range(0, len(sorted_sortino) / 2):
    k,v = sorted_sortino[i]
    sortino_set.add(k)



for code in jenson_set:
    if code in sortino_set:
        print code


