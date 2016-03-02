#coding=utf8


import string
from datetime import datetime
from datetime import timedelta


fund_dates = set()
index_dates = set()

f = open('data/funds','r')
lines = f.readlines()
f.close()

for line in lines:
	vec = line.strip().split()
	fund_dates.add(vec[1].strip())




f = open('./data/index_weekly.csv','r')
lines = f.readlines()
f.close()

for i in range(1, len(lines)):
        line = lines[i].strip()
        vec = line.strip().replace('/','-').split(',')
        d = vec[0].strip()
        d = (datetime.strptime(d, '%Y-%m-%d') + timedelta(2)).strftime('%Y-%m-%d')
	index_dates.add(d)	


final_dates = []
for d in index_dates:
	if d in fund_dates:
		final_dates.append(d)

final_dates.sort()

date_str = ''
for d in final_dates:
	date_str = date_str + d + ','
date_str = date_str[0 : len(date_str) - 1].strip()

print date_str
