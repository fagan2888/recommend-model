#coding=utf8

import string
import MySQLdb
import datetime

fund_infos = {}

conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='mofang', charset='utf8')

cur = conn.cursor()
cur.execute("select fi_globalid, fi_code, fi_name from fund_infos")
records = cur.fetchall()
for record in records:
	fund_infos[record[0]] = record
	#print record[0], record[1], record[2]

cur.close()
conn.commit()

fund_values = {}
cur = conn.cursor()
for key in fund_infos.keys():
	cur.execute("select fv_time, fv_net_value, fv_total_value, fv_authority_value from fund_value where fv_fund_id = '%d'" % (key))
	record = cur.fetchall()
	fund_values[key] = record


cur.close()
conn.commit()
conn.close()


conn = MySQLdb.connect(host='rdsf4ji381o0nt6n2954.mysql.rds.aliyuncs.com', port=3306, user='data', passwd='oDCiT8OfKaHf2E6P2cR3', db='data', charset='utf8')


fund_info_sql = "replace into fund_info (code, name, start_date, created_at, updated_at) values (%d, '%s', '%s', '%s', '%s')"
fund_value_sql = "replace into fund_value_daily (code, date, net_value, authority_value, net_profit, authority_profit ,created_at, updated_at ) values (%d, '%s', %f, %f, %f, %f, '%s', '%s')"


fund_mofang_ids = fund_infos.keys()
for key in fund_mofang_ids:

	info = fund_infos[key]
	fi_code = info[1]	
	fi_name = info[2]
	records = fund_values[key]	

	if fi_name.find('a') >= 0 or fi_name.find('A') >= 0 or fi_name.find('c') >= 0 or fi_name.find('C') >= 0 or fi_name.find('B') >= 0 or fi_name.find('b') >= 0 or fi_name.find(u'\u5206\u7EA7') >= 0:
		continue
			
	values = {}
	for record in records:
		#print record
		values[record[0]] = record
	dates = values.keys()
	dates.sort()


	if(len(dates) <= 0):
		continue

	cur = conn.cursor()
	#print fund_info_sql % (fi_code, fi_name, dates[0].strftime('%Y-%m-%d'), datetime.datetime.now(), datetime.datetime.now()) 
	cur.execute(fund_info_sql % (fi_code, fi_name, dates[0].strftime('%Y-%m-%d'), datetime.datetime.now(), datetime.datetime.now()))			

	pre_net_v = -1;	
	pre_authority_v = -1;


	for date in dates:
		record = values[date]
		net_value = record[1]		
		authority_value = record[3]	

		net_profit = 0.0;
		authority_profit = 0.0;

		if net_value <= 0.0 or authority_value <= 0.0:
			continue

		if pre_net_v > 0.0 and pre_authority_v > 0.0:
			net_profit = net_value / pre_net_v - 1;
			authority_profit = authority_value / pre_authority_v - 1;	

			if net_profit <= -1.0 or authority_profit <= -1.0:
				continue
				
	
		#vs.append( (fi_code, date, net_value, total_value, authority_value, 0.0, datetime.datetime.now(), datetime.datetime.now() ) )
		print fund_value_sql % (fi_code, date, net_value, authority_value, net_profit, authority_profit, datetime.datetime.now(), datetime.datetime.now())

		cur.execute(fund_value_sql % (fi_code, date, net_value, authority_value, net_profit, authority_profit, datetime.datetime.now(), datetime.datetime.now()))

		pre_net_v = net_value;	
		pre_authority_v = authority_value;
	
	#cur.executemany(fund_value_sql, vs)

	conn.commit()
	cur.close()


conn.close()
