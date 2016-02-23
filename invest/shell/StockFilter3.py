#coding=utf8

import string
import numpy as np
from datetime import datetime
import sys

lines = open(sys.argv[2],'r').readlines()
line = lines[0].strip()
items = line.split(',')
final_dates = []
for i in range(1, len(items) - 1):
	final_dates.append(datetime.strptime(items[i].strip(),'%Y-%m-%d'))
#print len(final_dates)

final_dates_set = set(final_dates)

lines = open(sys.argv[1],'r').readlines()

line = lines[0]


dates = []
for d in line.strip().split(','):
	dates.append(datetime.strptime(d, '%Y/%m/%d'))


'''
juchao_dates = []
for i in range(1, len(items) - 1):
	juchao_dates.append(datetime.strptime(items[i].strip(),'%Y-%m-%d'))
print juchao_dates
'''


shangzheng_index = []
items = lines[1].strip().split(',')
for i in range(0, len(items)):
	if dates[i] in final_dates_set:
		shangzheng_index.append(items[i])



shangzheng = []
for i in range(3,len(shangzheng_index)):
	n = 0
	for j in range(0,4):
		v = string.atof(shangzheng_index[i-j])
		if v >= 0:
			n = n + 1
		else:
			n = n - 1

	if n == 0:
		shangzheng.append(0)
	if n > 0:
		shangzheng.append(1)
	if n < 0:
		shangzheng.append(-1)



dapan_vec = lines[2].strip().split(',')
xiaopan_vec = lines[3].strip().split(',')



daxiaopan_index = []
for i in range(0, len(dapan_vec)):
	if dates[i] in final_dates_set:
		daxiaopan_index.append(string.atof(dapan_vec[i]) - string.atof(xiaopan_vec[i]))


daxiaopan = []
for i in range(3,len(daxiaopan_index)):
	n = 0
	for j in range(0,4):
		v = string.atof(daxiaopan_index[i-j])
		if v >= 0:
			n = n + 1
		else:
			n = n - 1
	if n == 0:
		daxiaopan.append(0)
	if n > 0:
		daxiaopan.append(1)
	if n < 0:
		daxiaopan.append(-1)



#print shangzheng
#print daxiaopan


chengzhang_vec = lines[4].strip().split(',')
jiazhi_vec = lines[5].strip().split(',')



chengzhangjiazhi_index = []
for i in range(0, len(chengzhang_vec)):
	if dates[i] in final_dates_set:
		chengzhangjiazhi_index.append(string.atof(chengzhang_vec[i]) - string.atof(jiazhi_vec[i]))


chengzhangjiazhi = []
for i in range(3,len(chengzhangjiazhi_index)):
	n = 0
	for j in range(0,4):
		v = string.atof(chengzhangjiazhi_index[i-j])
		if v >= 0:
			n = n + 1
		else:
			n = n - 1
	if n == 0:
		chengzhangjiazhi.append(0)
	if n > 0:
		chengzhangjiazhi.append(1)
	if n < 0:
		chengzhangjiazhi.append(-1)



#print shangzheng
#print daxiaopan
#print chengzhangjiazhi


rf = (0.015 / 52)
lable = {}
fundup = {}
funddown = {}
fundmiddle = {}
funddapan = {}
fundxiaopan = {}
fundchengzhang = {}
fundjiazhi = {}
fundupstd = {}
funddownstd = {}
fundmiddlestd = {}
funddapanstd = {}
fundxiaopanstd = {}
fundchengzhangstd = {}
fundjiazhistd = {}
lines = open(sys.argv[2],'r').readlines()
for i  in range(1, len(lines)):
	vec = lines[i].strip().split(',')	
	code = string.atoi(vec[0].strip())
	rs = []				
	for j in range(4, len(vec) - 1):
		rs.append(string.atof(vec[j].strip()))
	
	up       = []
	down     = []
	middle   = []


	dapan    = []
	xiaopan  = []
	
	chengzhang = []
	jiazhi     = []


	for j in range(0, len(shangzheng)):
		p = shangzheng[j]
		if p == 1:
			up.append(rs[j])	
		elif p == 0:
			middle.append(rs[j])
		else:
			down.append(rs[j])


	for j in range(0, len(daxiaopan)):
		p = daxiaopan[j]
		if p >= 0:
			dapan.append(rs[j])				
		else:
			xiaopan.append(rs[j])
		

	for j in range(0, len(chengzhangjiazhi)):
		p = chengzhangjiazhi[j]
		if p >= 0 :
			chengzhang.append(rs[j])
		else:
			jiazhi.append(rs[j])

		
	#print code , 'up' , up
	#print code , 'down', down
	#print code , 'middle', middle	
	#print code , 'dapan' , dapan
	#print code , 'xiaopan', xiaopan
	#print code , 'chengzhang', chengzhang
	#print code , 'jiazhi'	,  jiazhi

	
	print code , 'up' , (np.mean(up) - rf) /  np.std(up)
	print code , 'down', (np.mean(down) - rf) / np.std(down)
	print code , 'middle', (np.mean(middle) - rf) / np.std(middle)	
	print code , 'dapan' , (np.mean(dapan) - rf) / np.std(dapan)
	print code , 'xiaopan', (np.mean(xiaopan) - rf) / np.std(xiaopan)
	print code , 'chengzhang', (np.mean(chengzhang) - rf) / np.std(chengzhang)
	print code , 'jiazhi'	,  (np.mean(jiazhi) - rf) / np.std(jiazhi)
		
	fundup[code] = np.mean(up)
	funddown[code] = np.mean(down)
	fundmiddle[code] = np.mean(middle)
	funddapan[code] = np.mean(dapan)
	fundxiaopan[code] = np.mean(xiaopan)
	fundchengzhang[code] = np.mean(chengzhang)
	fundjiazhi[code] = np.mean(jiazhi)

	fundupstd[code] = np.std(up)
	funddownstd[code] = np.std(down)
	fundmiddlestd[code] = np.std(middle)
	funddapanstd[code]  = np.std(dapan)
	fundxiaopanstd[code] = np.std(xiaopan)
	fundchengzhangstd[code] = np.std(chengzhang)
	fundjiazhistd[code]     = np.std(jiazhi)		

	
#print shangzheng
#print daxiaopan
#print chengzhangjiazhi


