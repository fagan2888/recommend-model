#coding=utf8

import MySQLdb


infos = {}
dates = set()
conn = MySQLdb.connect(host='182.92.214.1', port=3306, user='jiaoyang', passwd='Mofang123', db='hedgefund', charset='utf8')
cur = conn.cursor()
cur.execute("select S_INFO_WINDCODE, TRADE_DT, F_AVGRETURN_THISWEEK from CHINAHEDGEFUNDPERFORMANCE")
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
	print key,",",
	for (d, r) in infos[key]:
		dv[d] = r
	ds = list(dv.keys())	
	ds.sort()
	for d in ds:
		print d,',',
	print
	print key,',',
	for d in ds:
		print dv[d],',',
	print	
	print 
