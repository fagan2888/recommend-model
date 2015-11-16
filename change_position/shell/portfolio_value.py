#coding=utf8

import MySQLdb
import datetime
import string

conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='recommend', charset='utf8')
'''
cur = conn.cursor()

portfolio_base_sql = "insert into portfolios (p_name, p_risk, pf_focus, pf_position_record, pf_anaual_returns, pf_expect_returns_max, pf_expect_returns_min, created_at, updated_at) values ('%s', '%f', '%s', '%s', '%f', '%f', '%f', '%s', '%s')"

sql = portfolio_base_sql % ('risk_one_20150506', 0.1, 'focus', 'position_record', 0.07, 0.10,0.05,'2015-05-06', '2015-05-06')
#print sql
cur.execute(sql)
sql = portfolio_base_sql % ('risk_one_20150811', 0.1, 'focus', 'position_record', 0.08, 0.11,0.06,'2015-08-11', '2015-08-11')
#print sql
cur.execute(sql)
sql = portfolio_base_sql % ('risk_one_20151116', 0.1, 'focus', 'position_record', 0.065, 0.12,0.05,datetime.datetime.now(), datetime.datetime.now())
#print sql
cur.execute(sql)
cur.close()
conn.commit()

'''

cur = conn.cursor()
weight_base_sql = "insert into portfolio_weights (pw_portfolio_id, pw_portfolio_name, pw_fund_id, pw_weight, pw_risk, created_at, updated_at) values ('%d', '%s', '%d', '%f', '%f', '%s', '%s')"
sql = weight_base_sql % (4, 'risk_one',30000621, 0.2, 0.1, '2015-05-06', '2015-05-06')
print sql
cur.execute(sql)
sql = weight_base_sql % (4, 'risk_one',30000858, 0.2, 0.1, '2015-05-06', '2015-05-06')
print sql
cur.execute(sql)
sql = weight_base_sql % (4, 'risk_one',30000696, 0.2, 0.1, '2015-05-06', '2015-05-06')
print sql
cur.execute(sql)
sql = weight_base_sql % (4, 'risk_one',30000471, 0.2, 0.1, '2015-05-06', '2015-05-06')
print sql
cur.execute(sql)
sql = weight_base_sql % (4, 'risk_one',30000340, 0.2, 0.1, '2015-05-06', '2015-05-06')
print sql
cur.execute(sql)


sql = weight_base_sql % (5, 'risk_one',30000621, 0.1, 0.1, '2015-08-11', '2015-08-11')
print sql
cur.execute(sql)
sql = weight_base_sql % (5, 'risk_one',30000858, 0.2, 0.1, '2015-08-11', '2015-08-11')
print sql
cur.execute(sql)
sql = weight_base_sql % (5, 'risk_one',30000696, 0.3, 0.1, '2015-08-11', '2015-08-11')
print sql
cur.execute(sql)
sql = weight_base_sql % (5, 'risk_one',30000471, 0.2, 0.1, '2015-08-11', '2015-08-11')
print sql
cur.execute(sql)
sql = weight_base_sql % (5, 'risk_one',30000340, 0.2, 0.1, '2015-08-11', '2015-08-11')
print sql
cur.execute(sql)

sql = weight_base_sql % (6, 'risk_one',30000621, 0.3, 0.1, datetime.datetime.now(), datetime.datetime.now())
print sql
cur.execute(sql)
sql = weight_base_sql % (6, 'risk_one',30000858, 0.1, 0.1, datetime.datetime.now(), datetime.datetime.now())
print sql
cur.execute(sql)
sql = weight_base_sql % (6, 'risk_one',30000696, 0.4, 0.1, datetime.datetime.now(), datetime.datetime.now())
print sql
cur.execute(sql)
sql = weight_base_sql % (6, 'risk_one',30000471, 0.1, 0.1, datetime.datetime.now(), datetime.datetime.now())
print sql
cur.execute(sql)
sql = weight_base_sql % (6, 'risk_one',30000340, 0.1, 0.1, datetime.datetime.now(), datetime.datetime.now())
print sql
cur.execute(sql)

cur.close()
conn.commit()


cur = conn.cursor()
value_base_sql = "insert into portfolio_weights (pw_portfolio_id, pw_portfolio_name, pw_fund_id, pw_weight, pw_risk, created_at, updated_at) values ('%d', '%s', '%d', '%f', '%f', '%s', '%s')"
values_base_sql = "insert into portfolio_values (pv_risk, pv_date, pv_value, pv_ratio, created_at, updated_at) values ('%f', '%s', '%f', '%f', '%s', '%s')"
f = open('./data/values.csv','r')
v = 0
for line in f.readlines():
	vec = line.strip().split(',')
	date = vec[0].strip()
	value = string.atof(vec[1].strip())
	if v == 0:
		v = value
	sql = values_base_sql % (0.1, date, value, value / v - 1, datetime.datetime.now(), datetime.datetime.now())
	print sql
	cur.execute(sql)
	v = value
cur.close()
conn.commit()


conn.close()
