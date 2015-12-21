#coding=utf8

import string
import datetime


data = {}

f = open('./data/zz500_index.utf8','r')
lines = f.readlines()

lines = lines[1:len(lines)]

for line in lines:
	vec = line.split(',')
	d = datetime.datetime.strptime(vec[0].strip(),'%Y/%m/%d')
	d_str = d.strftime('%Y-%m-%d')
	content = (string.atof(vec[1]), string.atof(vec[2]), string.atof(vec[3]), string.atof(vec[4]), string.atof(vec[5]), string.atof(vec[6]))
	data[d_str] = content
	


dates = data.keys()
dates.sort()

close = {}
for d in dates:
	close[d] = data[d][3]


mean_day = 20
ma20 = {}
for i in range(mean_day, len(dates)):
	ma = 0.0
	for j in range(1, mean_day + 1):
		ma = ma + close[dates[i - j]]	
	ma = 1.0 * ma / mean_day

	ma20[dates[i]] = ma


ratio = {}
for i in range(1, len(dates)):
	ratio[dates[i]]	= close[dates[i]] / close[dates[i - 1]] - 1


positions = {}

ds = ma20.keys()
ds.sort()


start_date = '2014-07-31'
end_date   = '2015-07-31' 

start_index = ds.index(start_date)
end_index   = ds.index(end_date)

ds = ds[start_index : end_index + 1]


flag = 0
threshold = 0.02
for d in ds:
	close_value = close[d]
	ma_value    = ma20[d]	
	positions[d] = flag
	if close_value >= ( 1 + threshold ) * ma_value:
		flag = 1	
	if close_value < ( 1 - threshold )  * ma_value:
		flag = 0	

value = 1
for d in ds:
	flag = positions[d]
	value = value * (1 + 1.0 * flag * ratio[d])
	print d, flag ,value
