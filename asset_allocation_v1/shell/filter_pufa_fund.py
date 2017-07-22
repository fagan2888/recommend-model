#coding=utf8


import sys
sys.path.append('shell')
import MySQLdb
import config
import pandas as pd
import re



if __name__ == '__main__':


    pattern = re.compile("[A-Za-z]")
    df = pd.read_csv('./data/all_pufa_fund.csv')
    codes = []
    for code in df.values:
        code = code[0]
        match = pattern.search(code)
        if not match:
            code = "%06d" % int(code)
            codes.append(code)
    codes = set(codes)


    conn  = MySQLdb.connect(**config.db_base)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)

    sql = 'select globalid, ra_code, ra_mask from ra_fund'
    df = pd.read_sql(sql, conn, index_col = ['globalid'])

    for gid in df.index:
        code = df.loc[gid, 'ra_code']
        if code not in codes:
            sql = 'update ra_fund set ra_mask = 1 where globalid = %d' % gid
            print sql
            cur.execute(sql)

    conn.close()
