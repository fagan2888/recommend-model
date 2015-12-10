#coding=utf8

import string
import MySQLdb



def datelist(start_date, end_date):

	date_list = []
	start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
	end_date   = datetime.datetime.strptime(end_date_str, '%Y-%m-%d') 
	delta = datetime.timedelta(days=1)
	d = start_date
	date_list.append(d.strftime('%Y-%m-%d'))
	while(d <= end_date):
	    d = d + delta
	    date_list.append(d.strftime('%Y-%m-%d')) 

	return date_list


conn = MySQLdb.connect(host='rdsf4ji381o0nt6n2954.mysql.rds.aliyuncs.com', port=3306, user='data', passwd='oDCiT8OfKaHf2E6P2cR3', db='data', charset='utf8')

cur = conn.cursor()

cur.execute('select code, date, net_value, authority_value from fund_value_daily')

fund_values = {}


for record in cur.fetchall():
	code = record[0]
	date = record[1]	
	net_value = record[2]
	authority_value = record[3]

		
