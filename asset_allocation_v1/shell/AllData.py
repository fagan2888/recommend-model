#coding=utf8


import string
import MySQLdb
from datetime import datetime
import pandas as pd
import numpy as np
import sys
sys.path.append('shell')
import Const
import dbconfig


db_params = dbconfig.db_mofang


def stock_fund_value():


    dates = set()
    nav_values_dict = {}            


    conn  = MySQLdb.connect(**db_params)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)


    #sql = "select a.* from (wind_fund_value a inner join wind_fund_type b on a.wf_fund_id=b.wf_fund_id ) inner join (select iv_time from index_value where iv_index_id =120000001 order by iv_time desc) as d  on d.iv_time=a.wf_time where b.wf_flag=1 and (b.wf_type like '20010101%%' or b.wf_type like '2001010201%%' or b.wf_type like '2001010202%%' or b.wf_type like '2001010204%%'  ) and b.wf_fund_code in (select fi_code from fund_infos where fi_regtime<='%s' and fi_regtime!='0000-00-00') and b.wf_fund_code not in (select wf_fund_code FROM wind_fund_type WHERE wf_end_time is not null and wf_end_time>='%s' and wf_type not like '20010101%%' and wf_type not like '2001010201%%' and wf_type not like '2001010202%%' and wf_type not like '2001010204%%') and a.wf_time>='%s' and a.wf_time<='%s'" % (start_date, end_date, start_date, end_date)

    sql = "select wf_fund_code, wf_nav_value, wf_time from wind_fund_value where wf_fund_code in (select wf_fund_code from wind_fund_type where wf_type like '20010101%%' or wf_type like '2001010201%%' or wf_type like '2001010202%%' or wf_type like '2001010204%%')"

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



if __name__ == '__main__':
    df = stock_fund_value()
    df.to_csv('stock.csv')
    print df


