#coding=utf8

import MySQLdb
import string


#30000621 
#get zz500 data
conn = MySQLdb.connect(host='182.92.214.1', port=3306, user='jiaoyang', passwd='Mofang123', db='mofang', charset='utf8')
cur = conn.cursor()
fid = 30000621
ret = cur.execute("select fv_time, fv_net_value from fund_value where fv_fund_id = %d" % (fid))   
dv = {}
for record in cur.fetchall():
	dv.setdefault(record[0], string.atof(record[1]))

cur.close()
conn.commit()
conn.close()


#get hs300_statuses
conn = MySQLdb.connect(host='182.92.214.1', port=3306, user='jiaoyang', passwd='Mofang123', db='recommend', charset='utf8')
cur = conn.cursor()
ret = cur.execute("select date, value from hs300_statuses")   
ds = []
for record in cur.fetchall():
	ds.append(record[0])

cur.close()
conn.commit()


cur = conn.cursor()
for d in ds:
	cur.execute("update hs300_statuses set value = %f where date = '%s'" % (dv[d], d))
	print "update hs300_statuses set value = %f where date = '%s'" % (dv[d], d)
cur.close()

conn.commit()
conn.close()
