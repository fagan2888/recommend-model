#coding=utf8


import sys
sys.path.append('shell')
import MySQLdb
import config
import pandas as pd
import re
from datetime import datetime


if __name__ == '__main__':

    f_path = sys.argv[1]
    df = pd.read_csv(f_path, index_col = ['date'], parse_dates = ['date'])


    conn  = MySQLdb.connect(**config.db_base)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)


    sql = 'select max(fi_globalid) from fund_infos'
    cur.execute(sql)
    record = cur.fetchone()
    gid = record['max(fi_globalid)']


    code_gid_dict = {}
    for code in df.columns:
        gid = gid + 1
        code_gid_dict[code] = gid
        begin_date = df.index[0]
        sql = "replace into fund_infos (fi_globalid, fi_code, fi_name, fi_regtime) values (%d, '%s', '%s', '%s')" % (gid, code, 'pfb_' + str(code), begin_date)
        print sql
        cur.execute(sql)



    nav_base_sql = "replace into caihui_spider_fund_value (cs_fund_id, cs_code, cs_time, cs_net_value, cs_total_value, cs_authority_value, cs_status, created_at, updated_at) values ('%d', '%s', '%s', '%f', '%f', '%f', 3, '%s', '%s')"
    for code in df.columns:
        gid = code_gid_dict[code]
        ser = df[code]
        for d in ser.index:
            v = ser.loc[d]
            sql = nav_base_sql % (gid, code, d, v, v, v, datetime.now(), datetime.now())
            print sql
            cur.execute(sql)

    conn.close()
