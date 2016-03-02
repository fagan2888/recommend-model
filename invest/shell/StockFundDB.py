#coding=utf8


import string
import MySQLdb
from datetime import datetime


#start_date = '2007-01-05'


#start_date = '2008-01-04'


#start_date = '2009-01-09'


#start_date = '2010-01-08'


#start_date = '2011-01-07'

start_date = '2012-01-06'

conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='mofang', charset='utf8')


#所有股票型基金
types = [5, 6, 9, 10, 15]
#types = [5, 9, 10, 15]
item_ids = []
for type_id in types:
	cur = conn.cursor()
	cur.execute("select it_item_id from item_type where it_type_id = %d" % type_id)
	records = cur.fetchall()
	for record in records:
		item_ids.append(record[0])
	cur.close()

conn.commit() 


fund_mofangids = []
#规模不超过100亿元，3年以上运作期的基金
base_date = datetime.strptime(start_date,'%Y-%m-%d')
for item in item_ids:
	cur = conn.cursor()
	cur.execute('select fi_regtime, fi_laste_raise ,fi_name from fund_infos where fi_globalid = %d' % item)	
	record = cur.fetchone()
	date = record[0]
	scale = record[1]


	#if record[2].strip().find(u'债') >= 0:
	#	continue

	if record[2].encode('utf8').find('债') >= 0:
		continue

	if date is None:
		continue
	if scale < 1000000:
		continue
	if scale >10000000000:
		continue

	date_str = date.strftime('%Y-%m-%d')
	raise_date = datetime.strptime(date_str, '%Y-%m-%d')

	if raise_date > base_date:
		continue

	fund_mofangids.append(item)
	cur.close()


conn.commit()	

#print fund_mofangids


funds = []
for mofangid in fund_mofangids:
	cur = conn.cursor()
	cur.execute('select fi_code from fund_infos where fi_globalid = %d' % mofangid)	
	record = cur.fetchone()
	funds.append(record[0])

	cur.close()

conn.commit()
conn.close()


conn = MySQLdb.connect(host='rdsf4ji381o0nt6n2954.mysql.rds.aliyuncs.com', port=3306, user='data', passwd='oDCiT8OfKaHf2E6P2cR3', db='data', charset='utf8')

for code in funds:
		
	cur = conn.cursor()
	cur.execute('select date, authority_value, authority_profit from fund_value_weekly where code = %d' % code)	
	records = cur.fetchall()
	for record in records:
		print code, record[0], record[1], record[2]

	cur.close()


conn.commit()
conn.close()
