#coding=utf8


import string
import MySQLdb
import config
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('shell')
import Const

from Const import datapath
from dateutil.parser import parse

def all_trade_dates():

    conn  = MySQLdb.connect(**config.db_base)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)

    sql = "SELECT td_date FROM trade_dates WHERE td_date >= '2002-01-04' AND td_type & 0x02 ORDER by td_date ASC";
    
    # sql = 'select iv_time from (select iv_time,DATE_FORMAT(`iv_time`,"%Y%u") week from (select * from index_value where iv_index_id =120000001  order by iv_time desc) as a group by iv_index_id,week order by week asc) as b'


    dates = []
    cur.execute(sql)
    records = cur.fetchall()
    for record in records:
        dates.append(record.values()[0].strftime('%Y-%m-%d'))
    conn.close()

    return dates



def trade_dates(start_date, end_date):
    conn  = MySQLdb.connect(**config.db_base)

    if not end_date:
        sql = "SELECT max(td_date) as td_date FROM trade_dates WHERE td_type & 0x02"
        end_date = db_pluck(conn, 'td_date', sql)

    if not end_date:
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)); 
        end_date = yesterday.strftime("%Y-%m-%d")

    sql = "SELECT td_date FROM trade_dates WHERE td_date BETWEEN '%s' AND '%s' AND (td_type & 0x02 OR td_date = '%s') ORDER By td_date ASC" % (start_date, end_date, end_date);

    df = pd.read_sql(sql, conn, index_col = 'td_date')
    conn.close()

    return df.index

    # conn  = MySQLdb.connect(**config.db_base)
    # cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    # conn.autocommit(True)

    # sql = "SELECT td_date FROM trade_dates WHERE td_date BETWEEN '%s' AND '%s' AND td_type & 0x02 ORDER By td_date ASC" % (start_date, end_date);


    # # sql = 'select iv_time from (select iv_time,DATE_FORMAT(`iv_time`,"%%Y%%u") week from (select * from index_value where iv_index_id =120000001 and iv_time >= "%s" and iv_time <= "%s" order by iv_time desc) as a group by iv_index_id,week order by week asc) as b' % (start_date, end_date)


    # dates = []
    # cur.execute(sql)
    # records = cur.fetchall()
    # for record in records:
    #     dates.append(record.values()[0].strftime('%Y-%m-%d'))
    # conn.close()

    # if '2016-10-14' not in dates:
    #     dates.append('2016-10-14')

    # dates.sort()
    
    # # dates = [x for x in dates if x != '2013-12-31' and x != '2012-12-31']

    # # print "todiff:%s,%s" % (start_date, end_date), dates
    
    
    # return dates

def db_pluck(conn, col, sql):
    cur = conn.cursor(MySQLdb.cursors.DictCursor)
    cur.execute(sql)
    records = cur.fetchall()

    if records:
        return records[0][col]
    return None

def trade_date_index(start_date, end_date=None):
    conn  = MySQLdb.connect(**config.db_base)

    if not end_date:
        sql = "SELECT max(td_date) as td_date FROM trade_dates WHERE td_date != CURDATE()"
        end_date = db_pluck(conn, 'td_date', sql)

    if not end_date:
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)); 
        end_date = yesterday.strftime("%Y-%m-%d").date()

    sql = "SELECT td_date as date FROM trade_dates WHERE td_date BETWEEN '%s' AND '%s' AND (td_type & 0x02 OR td_date = '%s') ORDER By td_date ASC" % (start_date, end_date, end_date);

    df = pd.read_sql(sql, conn, index_col = 'date', parse_dates=['date'])
    conn.close()

    return df.index

def trade_date_lookback_index(end_date=None, lookback=26, include_end_date=True):
    if include_end_date:
        condition = "(td_type & 0x02 OR td_date = '%s')" % (end_date)
    else:
        condition = "(td_type & 0x02)"
        
    sql = "SELECT td_date as date, td_type FROM trade_dates WHERE td_date <= '%s' AND %s ORDER By td_date DESC LIMIT %d" % (end_date, condition, lookback)

    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = 'date', parse_dates=['date'])
    conn.close()

    return df.index

def build_sql_trade_date_weekly(start_date, end_date, include_end_date=True):
    if include_end_date:
        condition = "(td_type & 0x02 OR td_date = '%s')" % (end_date)
    else:
        condition = "(td_type & 0x02)"

    return "SELECT td_date FROM trade_dates WHERE td_date BETWEEN '%s' AND '%s' AND %s" % (start_date, end_date, condition)

def build_sql_trade_date_daily(start_date, end_date):
    return "SELECT td_date FROM trade_dates WHERE td_date BETWEEN '%s' AND '%s'" % (start_date, end_date);


def stock_fund_value(start_date, end_date):

    dates = trade_date_index(start_date, end_date)

    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    # 按照周收盘取净值
    #
    date_sql = build_sql_trade_date_weekly(start_date, end_date)

    #
    # 按照基金类型筛选基金
    #
    type_sql = "SELECT DISTINCT wf_fund_id FROM wind_fund_type WHERE (wf_type like '20010101%%' OR wf_type like '2001010201%%' OR wf_type like '2001010202%%' OR wf_type like '2001010204%%') AND (wf_start_time <= '%s' AND (wf_end_time IS NULL OR wf_end_time >= '%s'))" % (end_date, end_date);
    #
    # 按照成立时间筛选基金
    #
    regtime_sql = "SELECT DISTINCT fi_globalid FROM fund_infos WHERE fi_regtime<='%s' and fi_regtime!='0000-00-00'" % (start_date);
    #
    # 使用inner jion 求交集
    #
    intersected = "SELECT B.wf_fund_id FROM (%s) AS B JOIN (%s) AS C ON B.wf_fund_id = C.fi_globalid" % (type_sql, regtime_sql);
    #
    #
    sql = "SELECT A.ra_date as date, A.ra_code as code, A.ra_nav_adjusted FROM ra_fund_nav A, (%s) D, (%s) E WHERE A.ra_fund_id = D.wf_fund_id AND A.ra_date = E.td_date ORDER BY A.ra_date" % (intersected, date_sql);
    
    # sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select c.iv_time from (select iv_time,DATE_FORMAT(`iv_time`,'%%Y%%u') week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as k group by iv_index_id,week order by week desc) as c) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '20010101%%' or b.wf_type like '2001010201%%' or b.wf_type like '2001010202%%' or b.wf_type like '2001010204%%'  ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '20010101%%' and wf_type not like '2001010201%%' and wf_type not like '2001010202%%' and wf_type not like '2001010204%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)


    print "stock_fund_value", sql
    
    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = ['date', 'code'], parse_dates=['date'])
    conn.close()

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df


def stock_day_fund_value(start_date, end_date):
    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    # 按照周收盘取净值
    #
    date_sql = build_sql_trade_date_daily(start_date, end_date)
    #
    # 按照基金类型筛选基金
    #
    type_sql = "SELECT DISTINCT wf_fund_id FROM wind_fund_type WHERE (wf_type like '20010101%%' OR wf_type like '2001010201%%' OR wf_type like '2001010202%%' OR wf_type like '2001010204%%') AND (wf_start_time <= '%s' AND (wf_end_time IS NULL OR wf_end_time >= '%s'))" % (end_date, end_date);
    #
    # 按照成立时间筛选基金
    #
    regtime_sql = "SELECT DISTINCT fi_globalid FROM fund_infos WHERE fi_regtime<='%s' and fi_regtime!='0000-00-00'" % (start_date);
    #
    # 使用inner jion 求交集
    #
    intersected = "SELECT B.wf_fund_id FROM (%s) AS B JOIN (%s) AS C ON B.wf_fund_id = C.fi_globalid" % (type_sql, regtime_sql);
    #
    #
    sql = "SELECT A.ra_date as date, A.ra_code as code, A.ra_nav_adjusted FROM ra_fund_nav A, (%s) D, (%s) E WHERE A.ra_fund_id = D.wf_fund_id AND A.ra_date = E.td_date ORDER BY A.ra_date" % (intersected, date_sql);

    # sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select iv_time from index_value where iv_index_id =120000001 order by iv_time desc) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '20010101%%' or b.wf_type like '2001010201%%' or b.wf_type like '2001010202%%' or b.wf_type like '2001010204%%'  ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '20010101%%' and wf_type not like '2001010201%%' and wf_type not like '2001010202%%' and wf_type not like '2001010204%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)

    #print sql
    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = ['date', 'code'], parse_dates=['date'])
    conn.close()

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df

def bond_fund_value(start_date, end_date):

    dates = trade_date_index(start_date, end_date)

    #sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select c.iv_time from (select iv_time,DATE_FORMAT(`iv_time`,'%%Y%%u') week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as k group by iv_index_id,week order by week desc) as c) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '2001010203%%' or b.wf_type like '20010103%%' ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010203%%' and wf_type not like '20010103%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)
    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    # 按照周收盘取净值
    #
    date_sql = build_sql_trade_date_weekly(start_date, end_date)
    #
    # 按照基金类型筛选基金
    #
    type_sql = "SELECT DISTINCT wf_fund_id FROM wind_fund_type WHERE (wf_type LIKE '2001010301%%' OR wf_type LIKE '2001010302%%' OR wf_type LIKE '2001010305%%') AND (wf_start_time <= '%s' AND (wf_end_time IS NULL OR wf_end_time >= '%s'))" % (end_date, end_date);
    #
    # 按照成立时间筛选基金
    #
    regtime_sql = "SELECT DISTINCT fi_globalid FROM fund_infos WHERE fi_regtime<='%s' and fi_regtime!='0000-00-00'" % (start_date);
    #
    # 使用inner jion 求交集
    #
    intersected = "SELECT B.wf_fund_id FROM (%s) AS B JOIN (%s) AS C ON B.wf_fund_id = C.fi_globalid" % (type_sql, regtime_sql);
    #
    #
    sql = "SELECT A.ra_date as date, A.ra_code as code, A.ra_nav_adjusted FROM ra_fund_nav A, (%s) D, (%s) E WHERE A.ra_fund_id = D.wf_fund_id AND A.ra_date = E.td_date ORDER BY A.ra_date" % (intersected, date_sql);
    
    # sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select c.iv_time from (select iv_time,DATE_FORMAT(`iv_time`,'%%Y%%u') week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as k group by iv_index_id,week order by week desc) as c) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '2001010301%%' or b.wf_type like '2001010302%%' or b.wf_type like '2001010305%%') and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010301%%' and wf_type not like '2001010302%%' and wf_type not like '2001010305') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)

    print "bond_fund_value", sql;

    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = ['date', 'code'], parse_dates=['date'])
    conn.close()

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df

def bond_day_fund_value(start_date, end_date):


    dates = set()

    #sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select iv_time from index_value where iv_index_id =120000001 order by iv_time desc) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '2001010203%%' or b.wf_type like '20010103%%' ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010203%%' and wf_type not like '20010103%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)

    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    # 按照周收盘取净值
    #
    date_sql = build_sql_trade_date_daily(start_date, end_date)
    #
    # 按照基金类型筛选基金
    #
    type_sql = "SELECT DISTINCT wf_fund_id FROM wind_fund_type WHERE (wf_type LIKE '2001010301%%' OR wf_type LIKE '2001010302%%' OR wf_type LIKE '2001010305%%') AND (wf_start_time <= '%s' AND (wf_end_time IS NULL OR wf_end_time >= '%s'))" % (end_date, end_date);
    #
    # 按照成立时间筛选基金
    #
    regtime_sql = "SELECT DISTINCT fi_globalid FROM fund_infos WHERE fi_regtime<='%s' and fi_regtime!='0000-00-00'" % (start_date);
    #
    # 使用inner jion 求交集
    #
    intersected = "SELECT B.wf_fund_id FROM (%s) AS B JOIN (%s) AS C ON B.wf_fund_id = C.fi_globalid" % (type_sql, regtime_sql);
    #
    #
    sql = "SELECT A.ra_date as date, A.ra_code as code, A.ra_nav_adjusted FROM ra_fund_nav A, (%s) D, (%s) E WHERE A.ra_fund_id = D.wf_fund_id AND A.ra_date = E.td_date ORDER BY A.ra_date" % (intersected, date_sql);

    # sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select iv_time from index_value where iv_index_id =120000001 order by iv_time desc) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '2001010301%%' or b.wf_type like '2001010302%%'  or b.wf_type like '2001010305%%') and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '2001010301%%' and wf_type not like '2001010302%%' and wf_type not like '2001010305%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)

    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = ['date', 'code'], parse_dates=['date'])
    conn.close()

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df


def money_fund_value(start_date, end_date):


    dates = trade_date_index(start_date, end_date)

    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    # 按照周收盘取净值
    #
    date_sql = build_sql_trade_date_weekly(start_date, end_date)
    #
    # 按照基金类型筛选基金
    #
    type_sql = "SELECT DISTINCT wf_fund_id FROM wind_fund_type WHERE (wf_type like '20010104%%') AND (wf_start_time <= '%s' AND (wf_end_time IS NULL OR wf_end_time >= '%s'))" % (end_date, end_date);
    #
    # 按照成立时间筛选基金
    #
    regtime_sql = "SELECT DISTINCT fi_globalid FROM fund_infos WHERE fi_regtime<='%s' and fi_regtime!='0000-00-00'" % (start_date);
    #
    # 使用inner jion 求交集
    #
    intersected = "SELECT B.wf_fund_id FROM (%s) AS B JOIN (%s) AS C ON B.wf_fund_id = C.fi_globalid" % (type_sql, regtime_sql);
    #
    #
    sql = "SELECT A.ra_date as date, A.ra_code as code, A.ra_nav_adjusted FROM ra_fund_nav A, (%s) D, (%s) E WHERE A.ra_fund_id = D.wf_fund_id AND A.ra_date = E.td_date ORDER BY A.ra_date" % (intersected, date_sql);

    # sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select c.iv_time from (select iv_time,DATE_FORMAT(`iv_time`,'%%Y%%u') week from (select * from index_value where iv_index_id =120000001 order by iv_time desc) as k group by iv_index_id,week order by week desc) as c) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '20010104%%' ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '20010104%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)

    print "money_fund_value", sql

    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = ['date', 'code'], parse_dates=['date'])
    conn.close()

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df



def money_day_fund_value(start_date, end_date):
    dates = set()

    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    # 按照周收盘取净值
    #
    date_sql = build_sql_trade_date_daily(start_date, end_date)
    #
    # 按照基金类型筛选基金
    #
    type_sql = "SELECT DISTINCT wf_fund_id FROM wind_fund_type WHERE (wf_type like '20010104%%') AND (wf_start_time <= '%s' AND (wf_end_time IS NULL OR wf_end_time >= '%s'))" % (end_date, end_date);
    #
    # 按照成立时间筛选基金
    #
    regtime_sql = "SELECT DISTINCT fi_globalid FROM fund_infos WHERE fi_regtime<='%s' and fi_regtime!='0000-00-00'" % (start_date);
    #
    # 使用inner jion 求交集
    #
    intersected = "SELECT B.wf_fund_id FROM (%s) AS B JOIN (%s) AS C ON B.wf_fund_id = C.fi_globalid" % (type_sql, regtime_sql);
    #
    #
    sql = "SELECT A.ra_date as date, A.ra_code as code, A.ra_nav_adjusted FROM ra_fund_nav A, (%s) D, (%s) E WHERE A.ra_fund_id = D.wf_fund_id AND A.ra_date = E.td_date ORDER BY A.ra_date" % (intersected, date_sql);

    # sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select iv_time from index_value where iv_index_id =120000001 order by iv_time desc) as d on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '20010104%%' ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '20010104%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)

    # print sql

    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = ['date', 'code'], parse_dates=['date'])
    conn.close()

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df




def index_value(start_date, end_date):

    dates = trade_date_index(start_date, end_date)

    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    date_sql = build_sql_trade_date_weekly(start_date, end_date)
    #
    #
    sql = "SELECT ra_date as date, ra_index_code, ra_nav FROM ra_index_nav, (%s) E WHERE ra_date = E.td_date ORDER BY ra_date" % (date_sql)
    
    # sql = "select iv_index_id,iv_index_code,iv_time,iv_value,DATE_FORMAT(`iv_time`,'%%Y%%u') week from ( select * from index_value where iv_time>='%s' and iv_time<='%s' order by iv_time desc) as k group by iv_index_id,week order by week desc" % (start_date, end_date)


    print "index_value", sql

    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = ['date', 'ra_index_code'], parse_dates=['date'])
    conn.close()

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df




def index_day_value(start_date, end_date):


    dates = set()

    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    date_sql = build_sql_trade_date_daily(start_date, end_date)
    #
    #
    sql = "SELECT ra_date as date, ra_index_code, ra_nav FROM ra_index_nav, (%s) E WHERE ra_date = E.td_date ORDER BY ra_date" % (date_sql)

    # sql = "select iv_index_id,iv_index_code,iv_time,iv_value from index_value where iv_time>='%s' and iv_time<='%s' " % (start_date, end_date)
    
    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = ['date', 'ra_index_code'], parse_dates=['date'])
    conn.close()
 
    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df


def other_fund_value(start_date, end_date):


    dates = trade_date_index(start_date, end_date)

    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    date_sql = build_sql_trade_date_weekly(start_date, end_date)

    #
    #
    sql = "SELECT ra_date as date, ra_index_code, ra_nav FROM ra_index_nav, (%s) E WHERE ra_date = E.td_date ORDER BY ra_date" % (date_sql)

    # sql = "select iv_index_id,iv_index_code,iv_time,iv_value,DATE_FORMAT(`iv_time`,'%%Y%%u') week from ( select * from index_value where iv_time>='%s' and iv_time<='%s' order by iv_time desc) as k group by iv_index_id,week order by week desc" % (start_date, end_date)

    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = ['date', 'ra_index_code'], parse_dates=['date'])
    conn.close()
 
    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    df = df[[Const.sp500_code, Const.gold_code, Const.hs_code]]

    return df

def other_day_fund_value(start_date, end_date):

    dates = set()

    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    date_sql = build_sql_trade_date_daily(start_date, end_date)
    #
    #
    sql = "SELECT ra_date as date, ra_index_code, ra_nav FROM ra_index_nav, (%s) E WHERE ra_date = E.td_date ORDER BY ra_date" % (date_sql)

    # sql = "select iv_index_id,iv_index_code,iv_time,iv_value from index_value where iv_time>='%s' and iv_time<='%s' " % (start_date, end_date)

    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = ['date', 'ra_index_code'], parse_dates=['date'])
    conn.close()
 
    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    df = df[[Const.sp500_code, Const.gold_code, Const.hs_code]]

    return df


def position():


    #dates = set()

    conn  = MySQLdb.connect(**config.db_base)
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


    conn  = MySQLdb.connect(**config.db_base)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)


    sql = "select a.fi_code,a.fi_laste_raise from fund_infos a inner join wind_fund_type b on a.fi_globalid=b.wf_fund_id  where b.wf_flag=1 and (b.wf_type like '20010101%%' or b.wf_type like '2001010201%%' or b.wf_type like '2001010202%%' or b.wf_type like '2001010204%%'  )"


    df = pd.read_sql(sql, conn, index_col = 'fi_code')

    #cur.execute(sql)

    #records = cur.fetchall()
    conn.close()

    return df



if __name__ == '__main__':

    #df = stock_fund_value('2014-01-03', '2016-06-03')
    #df = bond_fund_value('2014-01-03', '2016-06-03')
    #df = money_fund_value('2015-01-03', '2016-06-03')
    #df =  index_value('2010-01-28', '2016-06-03')
    #f =  other_fund_value('2014-01-03', '2016-06-03')
    #df =  position()
    #df =  scale()
    #df  = bond_day_fund_value('2014-01-03', '2016-06-03')
    #df  = bond_fund_value('2014-01-01', '2016-07-19')
    #df.to_csv(datapath('bond.csv'))
    print all_trade_dates()
    print trade_date_index('2014-01-03', '2016-06-03')
