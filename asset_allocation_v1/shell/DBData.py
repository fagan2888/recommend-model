#coding=utf8


import string
import MySQLdb
from datetime import datetime
import pandas as pd
import numpy as np


db_params = {
            "host": "127.0.0.1",
            "port": 3306,
            "user": "root",
            "passwd": "Mofang123",
            "db":"mofang",
            "charset": "utf8"
        }


def trade_dates():

	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)

	sql = 'select iv_time from (select iv_time,DATE_FORMAT(`iv_time`,"%Y%u") week from (select * from index_value where iv_index_id =120000001  order by iv_time desc) as a group by iv_index_id,week order by week desc) as b'

	dates = []
	cur.execute(sql)
	records = cur.fetchall()
	for record in records:
		dates.append(record.values()[0].strftime('%Y-%m-%d'))
	conn.close()

	return dates




def stock_fund_value(start_date, end_date):

	#type_codes = ['2001010101','2001010102','2001010103','2001010201','2001010202','2001010204']
	nav_values_dict = {}			

		
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)

	
	#sql = 'select a.wf_fund_code,a.wf_nav_value, a.wf_time from wind_fund_value a INNER join wind_fund_type b on a.wf_fund_id=b.wf_fund_id where b.wf_flag=1 and b.wf_start_time <= "%s" and b.wf_type like "%s%%" and b.wf_fund_id is not null and a.wf_time>="%s" and a.wf_time<="%s" and a.wf_time in (select iv_time from (select iv_time,DATE_FORMAT(`iv_time`,"%%Y%%u") week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as a group by iv_index_id,week order by week desc) as c) ORDER BY `wf_time` ASC' % (start_date, type_code, start_date, end_date)


	sql = "select a.wf_fund_code,a.wf_nav_value,a.wf_time from wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id WHERE b.wf_flag=1 and (b.wf_type like '2001010101%%' or b.wf_type like '2001010201%%' or b.wf_type like '2001010202%%' or b.wf_type like '2001010204%%' or b.wf_type like '2001010102%%' or b.wf_type like '2001010103%%' ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010101%%' and wf_type not like '2001010201%%' and wf_type not like '2001010202%%' and wf_type not like '2001010204%%' and wf_type not like '2001010102%%' and wf_type not like '2001010103%%' ) and a.wf_time>='%s' and a.wf_time<='%s' and a.wf_time in (select iv_time from (select iv_time,DATE_FORMAT(`iv_time`,'%%Y%%u') week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as k group by iv_index_id,week order by week desc) as c) ORDER BY `wf_time` ASC" % (start_date,start_date,start_date,end_date)

#	sql = "select a.* from wind_fund_type a inner join (select wf_fund_id FROM wind_fund_type WHERE wf_flag=1 and (wf_type like '2001010101%%' or wf_type like '2001010201%%' or wf_type like '2001010202%%' or wf_type like '2001010204%%' or wf_type like '2001010102%%' or wf_type like '2001010103%%' ) and wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010101%%' and wf_type not like '2001010201%%' and wf_type not like '2001010202%%' and wf_type not like '2001010204%%' and wf_type not like '2001010102%%' and wf_type not like '2001010103%%' ))" % (start_date,start_time)

	cur.execute(sql)

	records = cur.fetchall()

	for record in records:
		code      = record['wf_fund_code']
		codes.add(code)
		nav_value = record['wf_nav_value']
		date      = record['wf_time']
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = nav_value

	conn.close()

	nav_values = []	
	nav_codes  = []
	dates = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		ds = vs_dict.keys()
		ds.sort()
		dates = ds
		for d in ds:
			vs.append(vs_dict[d])	
		nav_values.append(vs)

	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	

	return df



def bond_fund_value(start_date, end_date):

	#type_codes = ['2001010101','2001010102','2001010103','2001010201','2001010202','2001010204']
	nav_values_dict = {}			

		
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)

	
	#sql = 'select a.wf_fund_code,a.wf_nav_value, a.wf_time from wind_fund_value a INNER join wind_fund_type b on a.wf_fund_id=b.wf_fund_id where b.wf_flag=1 and b.wf_start_time <= "%s" and b.wf_type like "%s%%" and b.wf_fund_id is not null and a.wf_time>="%s" and a.wf_time<="%s" and a.wf_time in (select iv_time from (select iv_time,DATE_FORMAT(`iv_time`,"%%Y%%u") week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as a group by iv_index_id,week order by week desc) as c) ORDER BY `wf_time` ASC' % (start_date, type_code, start_date, end_date)


	sql = "select a.wf_fund_code,a.wf_nav_value,a.wf_time from wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id WHERE b.wf_flag=1 and (b.wf_type like '2001010101%%' or b.wf_type like '2001010201%%' or b.wf_type like '2001010202%%' or b.wf_type like '2001010204%%' or b.wf_type like '2001010102%%' or b.wf_type like '2001010103%%' ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010101%%' and wf_type not like '2001010201%%' and wf_type not like '2001010202%%' and wf_type not like '2001010204%%' and wf_type not like '2001010102%%' and wf_type not like '2001010103%%' ) and a.wf_time>='%s' and a.wf_time<='%s' and a.wf_time in (select iv_time from (select iv_time,DATE_FORMAT(`iv_time`,'%%Y%%u') week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as k group by iv_index_id,week order by week desc) as c) ORDER BY `wf_time` ASC" % (start_date,start_date,start_date,end_date)

#	sql = "select a.* from wind_fund_type a inner join (select wf_fund_id FROM wind_fund_type WHERE wf_flag=1 and (wf_type like '2001010101%%' or wf_type like '2001010201%%' or wf_type like '2001010202%%' or wf_type like '2001010204%%' or wf_type like '2001010102%%' or wf_type like '2001010103%%' ) and wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010101%%' and wf_type not like '2001010201%%' and wf_type not like '2001010202%%' and wf_type not like '2001010204%%' and wf_type not like '2001010102%%' and wf_type not like '2001010103%%' ))" % (start_date,start_time)

	cur.execute(sql)

	records = cur.fetchall()

	for record in records:
		code      = record['wf_fund_code']
		codes.add(code)
		nav_value = record['wf_nav_value']
		date      = record['wf_time']
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = nav_value

	conn.close()

	nav_values = []	
	nav_codes  = []
	dates = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		ds = vs_dict.keys()
		ds.sort()
		dates = ds
		for d in ds:
			vs.append(vs_dict[d])	
		nav_values.append(vs)

	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	

	return df




def money_fund_value(start_date, end_date):

	#type_codes = ['2001010101','2001010102','2001010103','2001010201','2001010202','2001010204']
	nav_values_dict = {}			

		
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)

	
	#sql = 'select a.wf_fund_code,a.wf_nav_value, a.wf_time from wind_fund_value a INNER join wind_fund_type b on a.wf_fund_id=b.wf_fund_id where b.wf_flag=1 and b.wf_start_time <= "%s" and b.wf_type like "%s%%" and b.wf_fund_id is not null and a.wf_time>="%s" and a.wf_time<="%s" and a.wf_time in (select iv_time from (select iv_time,DATE_FORMAT(`iv_time`,"%%Y%%u") week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as a group by iv_index_id,week order by week desc) as c) ORDER BY `wf_time` ASC' % (start_date, type_code, start_date, end_date)


	sql = "select a.wf_fund_code,a.wf_nav_value,a.wf_time from wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id WHERE b.wf_flag=1 and (b.wf_type like '2001010101%%' or b.wf_type like '2001010201%%' or b.wf_type like '2001010202%%' or b.wf_type like '2001010204%%' or b.wf_type like '2001010102%%' or b.wf_type like '2001010103%%' ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010101%%' and wf_type not like '2001010201%%' and wf_type not like '2001010202%%' and wf_type not like '2001010204%%' and wf_type not like '2001010102%%' and wf_type not like '2001010103%%' ) and a.wf_time>='%s' and a.wf_time<='%s' and a.wf_time in (select iv_time from (select iv_time,DATE_FORMAT(`iv_time`,'%%Y%%u') week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as k group by iv_index_id,week order by week desc) as c) ORDER BY `wf_time` ASC" % (start_date,start_date,start_date,end_date)

#	sql = "select a.* from wind_fund_type a inner join (select wf_fund_id FROM wind_fund_type WHERE wf_flag=1 and (wf_type like '2001010101%%' or wf_type like '2001010201%%' or wf_type like '2001010202%%' or wf_type like '2001010204%%' or wf_type like '2001010102%%' or wf_type like '2001010103%%' ) and wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010101%%' and wf_type not like '2001010201%%' and wf_type not like '2001010202%%' and wf_type not like '2001010204%%' and wf_type not like '2001010102%%' and wf_type not like '2001010103%%' ))" % (start_date,start_time)

	cur.execute(sql)

	records = cur.fetchall()

	for record in records:
		code      = record['wf_fund_code']
		codes.add(code)
		nav_value = record['wf_nav_value']
		date      = record['wf_time']
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = nav_value

	conn.close()

	nav_values = []	
	nav_codes  = []
	dates = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		ds = vs_dict.keys()
		ds.sort()
		dates = ds
		for d in ds:
			vs.append(vs_dict[d])	
		nav_values.append(vs)

	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	

	return df








if __name__ == '__main__':

	#trade_dates()	
	stock_fund_value('2014-01-03', '2016-06-03')

