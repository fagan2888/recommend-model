#coding=utf8


import string
import MySQLdb
from datetime import datetime
import pandas as pd
import numpy as np
import sys
sys.path.append('shell')
import Const


db_params = {
            "host": "rdsijnrreijnrre.mysql.rds.aliyuncs.com",
            "port": 3306,
            "user": "koudai",
            "passwd": "Mofang123",
            "db":"mofang_api",
            "charset": "utf8"
        }



def all_trade_dates():

	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)

	sql = 'select iv_time from (select iv_time,DATE_FORMAT(`iv_time`,"%Y%u") week from (select * from index_value where iv_index_id =120000001  order by iv_time desc) as a group by iv_index_id,week order by week asc) as b'

	dates = []
	cur.execute(sql)
	records = cur.fetchall()
	for record in records:
		dates.append(record.values()[0].strftime('%Y-%m-%d'))
	conn.close()

	return dates



def trade_dates(start_date, end_date):

	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)

	sql = 'select iv_time from (select iv_time,DATE_FORMAT(`iv_time`,"%%Y%%u") week from (select * from index_value where iv_index_id =120000001 and iv_time >= "%s" and iv_time <= "%s" order by iv_time desc) as a group by iv_index_id,week order by week asc) as b' % (start_date, end_date)

	dates = []
	cur.execute(sql)
	records = cur.fetchall()
	for record in records:
		dates.append(record.values()[0].strftime('%Y-%m-%d'))
	conn.close()

	return dates


def stock_fund_value(start_date, end_date):


	dates = trade_dates(start_date, end_date)
	dates.sort()

	ds = []
	for d in dates:
		ds.append(datetime.strptime(d,'%Y-%m-%d').date())
	dates = ds

	nav_values_dict = {}			

		
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)


	sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select c.iv_time from (select iv_time,DATE_FORMAT(`iv_time`,'%%Y%%u') week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as k group by iv_index_id,week order by week desc) as c) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '20010101%%' or b.wf_type like '2001010201%%' or b.wf_type like '2001010202%%' or b.wf_type like '2001010204%%'  ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '20010101%%' and wf_type not like '2001010201%%' and wf_type not like '2001010202%%' and wf_type not like '2001010204%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)


	#print sql	
	cur.execute(sql)


	records = cur.fetchall()

	for record in records:
		code      = record['wf_fund_code']
		nav_value = record['wf_nav_value']
		date      = record['wf_time']
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = float(nav_value)

	conn.close()


	nav_values = []	
	nav_codes  = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		ds = vs_dict.keys()
		ds.sort()
		for d in dates:
			if vs_dict.has_key(d):
				vs.append(vs_dict[d])
			else:
				vs.append(np.NaN)

		nav_values.append(vs)
	

	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	
	df = df.fillna(method='pad')
	df.index.name = 'date'
	return df



def stock_day_fund_value(start_date, end_date):


	dates = set()
	nav_values_dict = {}			


	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)


	sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select iv_time from index_value where iv_index_id =120000001 order by iv_time desc) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '20010101%%' or b.wf_type like '2001010201%%' or b.wf_type like '2001010202%%' or b.wf_type like '2001010204%%'  ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '20010101%%' and wf_type not like '2001010201%%' and wf_type not like '2001010202%%' and wf_type not like '2001010204%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)

	#print sql	
	cur.execute(sql)

	records = cur.fetchall()

	for record in records:
		code      = record['wf_fund_code']
		nav_value = record['wf_nav_value']
		date      = record['wf_time']
		dates.add(date)
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = float(nav_value)

	conn.close()

	dates = list(dates)
	dates.sort()

	nav_values = []	
	nav_codes  = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		ds = vs_dict.keys()
		ds.sort()
		for d in dates:
			if vs_dict.has_key(d):
				vs.append(vs_dict[d])
			else:
				vs.append(np.NaN)

		nav_values.append(vs)
	

	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	
	df = df.fillna(method='pad')
	df.index.name = 'date'
	return df


def bond_fund_value(start_date, end_date):


	dates = trade_dates(start_date, end_date)
	dates.sort()
	ds = []
	for d in dates:
		ds.append(datetime.strptime(d,'%Y-%m-%d').date())
	dates = ds

	nav_values_dict = {}			

		
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)


	sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select c.iv_time from (select iv_time,DATE_FORMAT(`iv_time`,'%%Y%%u') week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as k group by iv_index_id,week order by week desc) as c) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '2001010203%%' or b.wf_type like '20010103%%' ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010203%%' and wf_type not like '20010103%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)


	
	cur.execute(sql)

	
	records = cur.fetchall()

	for record in records:
		code      = record['wf_fund_code']
		nav_value = record['wf_nav_value']
		date      = record['wf_time']
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = float(nav_value)

	conn.close()


	nav_values = []	
	nav_codes  = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		for d in dates:
			if vs_dict.has_key(d):
				vs.append(vs_dict[d])
			else:
				vs.append(np.NaN)

		nav_values.append(vs)
		#print code , len(vs)	

	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	

	df = df.fillna(method='pad')

	df.index.name = 'date'

	return df


def bond_day_fund_value(start_date, end_date):


	dates = set()

	nav_values_dict = {}			
		
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)


	sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select iv_time from index_value where iv_index_id =120000001 order by iv_time desc) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '2001010203%%' or b.wf_type like '20010103%%' ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010203%%' and wf_type not like '20010103%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)

	
	cur.execute(sql)

	
	records = cur.fetchall()
	for record in records:
		code      = record['wf_fund_code']
		nav_value = record['wf_nav_value']
		date      = record['wf_time']
		dates.add(date)
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = float(nav_value)

	conn.close()

	dates = list(dates)
	dates.sort()

	nav_values = []	
	nav_codes  = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		for d in dates:
			if vs_dict.has_key(d):
				vs.append(vs_dict[d])
			else:
				vs.append(np.NaN)

		nav_values.append(vs)
		#print code , len(vs)	

	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	

	df = df.fillna(method='pad')

	df.index.name = 'date'

	return df


def money_fund_value(start_date, end_date):


	dates = trade_dates(start_date, end_date)
	dates.sort()
	ds = []
	for d in dates:
		ds.append(datetime.strptime(d,'%Y-%m-%d').date())
	dates = ds

	nav_values_dict = {}			

		
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)


	sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select c.iv_time from (select iv_time,DATE_FORMAT(`iv_time`,'%%Y%%u') week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as k group by iv_index_id,week order by week desc) as c) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '20010104%%' ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '20010104%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)


	
	cur.execute(sql)

	
	records = cur.fetchall()

	for record in records:
		code      = record['wf_fund_code']
		nav_value = record['wf_nav_value']
		date      = record['wf_time']
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = float(nav_value)

	conn.close()


	nav_values = []	
	nav_codes  = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		for d in dates:
			if vs_dict.has_key(d):
				vs.append(vs_dict[d])
			else:
				vs.append(np.NaN)

		nav_values.append(vs)
		#print code , len(vs)	

	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	

	df = df.fillna(method='pad')

	df.index.name = 'date'

	return df



def money_day_fund_value(start_date, end_date):


	dates = set()

	nav_values_dict = {}			
		
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)

	sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select iv_time from index_value where iv_index_id =120000001 order by iv_time desc) as d on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '20010104%%' ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '20010104%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)


	cur.execute(sql)
	
	records = cur.fetchall()

	for record in records:
		code      = record['wf_fund_code']
		nav_value = record['wf_nav_value']
		date      = record['wf_time']
		dates.add(date)
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = float(nav_value)

	conn.close()


	dates = list(dates)
	dates.sort()

	nav_values = []	
	nav_codes  = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		for d in dates:
			if vs_dict.has_key(d):
				vs.append(vs_dict[d])
			else:
				vs.append(np.NaN)

		nav_values.append(vs)
		#print code , len(vs)	

	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	

	df = df.fillna(method='pad')

	df.index.name = 'date'

	return df




def index_value(start_date, end_date):


	dates = trade_dates(start_date, end_date)
	dates.sort()
	ds = []
	for d in dates:
		ds.append(datetime.strptime(d,'%Y-%m-%d').date())
	dates = ds

	nav_values_dict = {}			

		
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)


	sql = "select iv_index_id,iv_index_code,iv_time,iv_value,DATE_FORMAT(`iv_time`,'%%Y%%u') week from ( select * from index_value where iv_time>='%s' and iv_time<='%s' order by iv_time desc) as k group by iv_index_id,week order by week desc" % (start_date, end_date)


	#print sql
	cur.execute(sql)

	
	records = cur.fetchall()

	for record in records:
		code      = record['iv_index_code']
		nav_value = record['iv_value']
		date      = record['iv_time']
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = float(nav_value)

	conn.close()


	nav_values = []	
	nav_codes  = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		for d in dates:
			if vs_dict.has_key(d):
				vs.append(vs_dict[d])
			else:
				vs.append(np.NaN)

		nav_values.append(vs)
		#print code , len(vs)	

	#print nav_values
	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	

	df = df.fillna(method='pad')

	df.index.name = 'date'

	return df



def index_day_value(start_date, end_date):


	dates = set()

	nav_values_dict = {}			
		
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)


	sql = "select iv_index_id,iv_index_code,iv_time,iv_value from index_value where iv_time>='%s' and iv_time<='%s' " % (start_date, end_date)
	cur.execute(sql)
	
	records = cur.fetchall()

	for record in records:
		code      = record['iv_index_code']
		nav_value = record['iv_value']
		date      = record['iv_time']
		dates.add(date)
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = float(nav_value)

	conn.close()


	dates = list(dates)
	dates.sort()

	nav_values = []	
	nav_codes  = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		for d in dates:
			if vs_dict.has_key(d):
				vs.append(vs_dict[d])
			else:
				vs.append(np.NaN)

		nav_values.append(vs)
		#print code , len(vs)	

	#print nav_values
	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	

	df = df.fillna(method='pad')

	df.index.name = 'date'

	return df



def other_fund_value(start_date, end_date):


	dates = trade_dates(start_date, end_date)
	dates.sort()
	ds = []
	for d in dates:
		ds.append(datetime.strptime(d,'%Y-%m-%d').date())
	dates = ds

	nav_values_dict = {}			

		
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)


	sql = "select iv_index_id,iv_index_code,iv_time,iv_value,DATE_FORMAT(`iv_time`,'%%Y%%u') week from ( select * from index_value where iv_time>='%s' and iv_time<='%s' order by iv_time desc) as k group by iv_index_id,week order by week desc" % (start_date, end_date)

	
	cur.execute(sql)

	
	records = cur.fetchall()

	for record in records:
		code      = record['iv_index_code']
		nav_value = record['iv_value']
		date      = record['iv_time']
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = float(nav_value)

	conn.close()


	nav_values = []	
	nav_codes  = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		for d in dates:
			if vs_dict.has_key(d):
				vs.append(vs_dict[d])
			else:
				vs.append(np.NaN)

		nav_values.append(vs)
		#print code , len(vs)	

	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	

	df = df.fillna(method='pad')

	df = df[[Const.sp500_code, Const.gold_code, Const.hs_code]]

	df.index.name = 'date'

	return df


def other_day_fund_value(start_date, end_date):

	dates = set()

	nav_values_dict = {}			

	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)


	sql = "select iv_index_id,iv_index_code,iv_time,iv_value from index_value where iv_time>='%s' and iv_time<='%s' " % (start_date, end_date)

	
	cur.execute(sql)

	
	records = cur.fetchall()

	for record in records:
		code      = record['iv_index_code']
		nav_value = record['iv_value']
		date      = record['iv_time']
		vs = nav_values_dict.setdefault(code, {})
		vs[date]  = float(nav_value)

	conn.close()


	nav_values = []	
	nav_codes  = []
	for code in nav_values_dict.keys():
		nav_codes.append(code)
		vs = []
		vs_dict = nav_values_dict[code]	
		for d in dates:
			if vs_dict.has_key(d):
				vs.append(vs_dict[d])
			else:
				vs.append(np.NaN)

		nav_values.append(vs)
		#print code , len(vs)	

	df = pd.DataFrame(np.matrix(nav_values).T, index = dates, columns = nav_codes)	

	df = df.fillna(method='pad')

	df = df[[Const.sp500_code, Const.gold_code, Const.hs_code]]

	df.index.name = 'date'

	return df



def position():


	#dates = set()
	
	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)


	sql = "select wf_fund_code, wf_time, wf_stock_value from wind_fund_stock_value"
	
	cur.execute(sql)
	records = cur.fetchall()

	position_dict = {}

	code_position = {}	


	for record in records:

		code      = record['wf_fund_code']
		position  = record['wf_stock_value']
		ps        = code_position.setdefault(code, [])
		ps.append(position)

	conn.close()

	for code in code_position.keys():
		position_dict[code] = np.mean(code_position[code])


	dates = list(dates)
	dates.sort()

	position_values = []	
	position_codes  = []
	for code in position_dict.keys():
		position_codes.append(code)
		ps = []
		ps_dict = position_dict[code]	
		for d in dates:
			if ps_dict.has_key(d):
				ps.append(ps_dict[d])
			else:
				ps.append(np.NaN)


		position_values.append(ps)
		#print code , len(vs)	


	df = pd.DataFrame(np.matrix(position_values).T, index = dates, columns = position_codes)	
	df = df.fillna(method='pad')
	df.index.name = 'date'
	return df



def scale():


	conn  = MySQLdb.connect(**db_params)
	cur   = conn.cursor(MySQLdb.cursors.DictCursor)
	conn.autocommit(True)


	sql = "select a.fi_code,a.fi_laste_raise from fund_infos a inner join wind_fund_type b on a.fi_globalid=b.wf_fund_id  where b.wf_flag=1 and (b.wf_type like '20010101%%' or b.wf_type like '2001010201%%' or b.wf_type like '2001010202%%' or b.wf_type like '2001010204%%'  )"


	df = pd.read_sql(sql, conn, index_col = 'fi_code')

	#cur.execute(sql)

	#records = cur.fetchall()
	conn.close()

	return df	



if __name__ == '__main__':

	#trade_dates()	
	#df = stock_fund_value('2014-01-03', '2016-06-03')
	#df = bond_fund_value('2014-01-03', '2016-06-03')
	#df = money_fund_value('2015-01-03', '2016-06-03')
	#df =  index_value('2010-01-28', '2016-06-03')
	#f =  other_fund_value('2014-01-03', '2016-06-03')
	#df =  position()
	#df =  scale()
	#df  = bond_day_fund_value('2014-01-03', '2016-06-03')
	df  = stock_fund_value('2015-01-01', '2016-06-03')
	print df
	dfr = df.pct_change().fillna(0.0)
	dfr.to_csv('./tmp/stock.csv')
	#print trade_dates('2014-01-03', '2016-06-03')

