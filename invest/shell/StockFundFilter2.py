#coding=utf8


import sys
sys.path.append("shell")
import string
import Financial as fin
import numpy as np
from datetime import datetime
from datetime import timedelta


lines = open(sys.argv[1],'r').readlines()


funds = {}
dates = set()
for line in lines:
	vec = line.split()
	code = vec[0]
	date = datetime.strptime(vec[1], '%Y-%m-%d')
	dates.add(date)
	r    = vec[3]
	rs   = funds.setdefault(code, {})
	rs[date] = r


'''
print ',',
for d in dates:
	print d,',',
print 
for code in funds.keys():
	rs = funds[code]
	print code, ',',
	for d in dates:
		if rs.has_key(d):
			print rs[d],',',
		else:
			print ',',
	print
		
'''


index = {}
final_dates = []
indexlines = open(sys.argv[2], 'r').readlines()
for line in indexlines:
	line = line.strip()
	#print line
	vec = line.split(',')
	d = datetime.strptime(vec[0], '%Y-%m-%d')
	d = d + timedelta(2)	
	if d in dates:
		final_dates.append(d)

	index[d] = string.atof(vec[1])
	#indexr.append(string.atof(line))		


final_dates.sort()
#print dates


n = len(final_dates)
m = len(funds.keys())
fundr = np.zeros((n,m))

codes = funds.keys()
for code in codes:
	for k,v in funds[code].items():
		try:	
			n = final_dates.index(k)
			m = codes.index(code)
			fundr[n][m] = v
		except:
			pass



#print fundr
indexr = []
for date in final_dates:
	indexr.append(index[date])	
	


rf = 0.015 / 52
jensen = {}
sortino = {}
for i in range(0, len(codes)):
	rs = []
	for j in range(0, len(final_dates)):
		rs.append(fundr[j][i])	 
	jensen[codes[i]] = fin.jensen(rs, indexr, rf)
	sortino[codes[i]] = fin.sortino(rs, rf)
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

final_codes = []

for code in jenson_set:
	if code in sortino_set:
		final_codes.append(code)	

print ',',
for date in final_dates:
	print date.strftime('%Y-%m-%d'),',',
print
i	
 
for code in final_codes:
	print code,',',
	i = codes.index(code)
	for j in range(0, len(final_dates)):
		print fundr[j][i],',',
	print	


#print indexr
#print len(indexr)
#print len(fundr)
#for d in dates:
#	print d
#print indexr
#print fundr
#print len(funds.keys())
#for code in funds.keys():
#	print len(funds[code])

