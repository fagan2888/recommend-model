#coding=utf8

import MySQLdb
import json
import datetime
import sys
reload(sys)
sys.setdefaultencoding('utf8')


infos = {}
conn = MySQLdb.connect(host='rdsijnrreijnrre.mysql.rds.aliyuncs.com', port=3306, user='koudai', passwd='Mofang123', db='recommend', charset='utf8')
cur = conn.cursor()
cur.execute("select ur_uid, ur_date, ur_risk from user_risk_analyze_results")
records = cur.fetchall()
for record in records:
	uid = record[0].strip()
	if(uid.find('4') == 0):
		continue
	d = record[1]
	risk_grade = record[2]
	info = infos.setdefault(uid,[])
	if len(info) == 0:
		info.append(d)
		info.append(risk_grade)
	else:
		info[0] = d
		info[1] = risk_grade
cur.close()
conn.commit()
conn.close()

#print infos



conn = MySQLdb.connect(host='rdsijnrreijnrre.mysql.rds.aliyuncs.com', port=3306, user='koudai', passwd='Mofang123', db='passport', charset='utf8')
cur = conn.cursor()
sql = "select mobile, created_at from users where id = %s"
for uid in infos.keys():
	cur.execute(sql % uid)
	r = cur.fetchone()
	if r == None:
		continue
	info = infos[uid]
	info.append(r[0])
	info.append(r[1])
cur.close()
conn.commit()
conn.close()



conn = MySQLdb.connect(host='rdsijnrreijnrre.mysql.rds.aliyuncs.com', port=3306, user='koudai', passwd='Mofang123', db='trade', charset='utf8')
cur = conn.cursor()
sql = "select ha_name from howbuy_accounts where ha_uid = %s"
for uid in infos.keys():
        cur.execute(sql % uid)
        r = cur.fetchone()
        if r == None:
                continue
        info = infos[uid]
        info.append(r[0])
cur.close()
conn.commit()
conn.close()



conn = MySQLdb.connect(host='rdsijnrreijnrre.mysql.rds.aliyuncs.com', port=3306, user='koudai', passwd='Mofang123', db='trade', charset='utf8')
cur = conn.cursor()
sql = "select ht_name, ht_code, ht_trade_amt, ht_trade_date from howbuy_trade_statuses where ht_uid = %s and ht_trade_type = 1 and (ht_trade_status = 6 or ht_trade_status = 7)"
for uid in infos.keys():
        cur.execute(sql % uid)
	trades = []
        for r in cur.fetchall():
		trades.append(r)		
        info = infos[uid]
	info.append(trades)
cur.close()
conn.commit()
conn.close()


for key in infos.keys():
	print key,
	info = infos[key]
	#print info
	l = len(info)
	n = 0
	for item in info:
		n = n + 1
		if n == l:
			for v in item:
				print v[0],",", v[1],",", v[2],"," ,v[3],",",
			continue
		print item,',',
	print
