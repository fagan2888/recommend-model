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

    sql = 'select max(globalid) from ra_fund'
    cur.execute(sql)
    record = cur.fetchone()
    gid = record['max(globalid)']

    code_gid_dict = {}
    for code in df.columns:
        gid = gid + 1
        code_git_dict[code] = gid
        begin_date = df.index[0]
        sql = "replace into ra_fund (globalid, ra_code, ra_name, ra_type, ra_type_calc, ra_regtime, ra_mask, created_at, updated_at) values (%d, '%s', '%s', 2, 1, '%s', 0, '%s', '%s')" % (gid, code, 'pufa_bank_' + str(code), begin_date, datetime.now(), datetime.now())
        cur.execute(sql)



    conn.close()
