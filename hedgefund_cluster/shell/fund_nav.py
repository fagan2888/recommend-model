#coding=utf8

import MySQLdb
import datetime

infos = {}
dates = set()
conn = MySQLdb.connect(host='182.92.214.1', port=3306, user='jiaoyang', passwd='Mofang123', db='hedgefund', charset='utf8')
cur = conn.cursor()
cur.execute("select F_INFO_WINDCODE, PRICE_DATE, F_NAV_ACCUMULATED from CHINAHEDGEFUNDNAV")
for record in cur.fetchall():
	code = record[0]
	d    = record[1]
	dates.add(d)
	r    = record[2]
	info = infos.setdefault(code,[])
	info.append((d, r))
cur.close()
conn.commit()
conn.close()



for key in infos:
	dv = {}
	#print key,",",
	for (d, r) in infos[key]:
		dv[d] = r
	ds = list(dv.keys())	
	if len(ds) < 20:
		continue
	ds.sort()
	#for d in ds:
	#	print d,',',
	#print
	print key,',',
	for d in ds:
		print dv[d],',',
	print	




'''
print len(infos.keys())
for key in infos:
	i = 0
	items = infos[key]
						

print n
'''


