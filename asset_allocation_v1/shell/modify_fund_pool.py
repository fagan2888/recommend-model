#coding=utf8


import sys
sys.path.append('shell')
import MySQLdb
import config
import pandas as pd
import re
from datetime import datetime


if __name__ == '__main__':

    op        = sys.argv[1]
    pool_id   = sys.argv[2]
    fund_code = sys.argv[3]


    conn  = MySQLdb.connect(**config.db_base)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)


    sql = 'select globalid from ra_fund where ra_code = %s' % fund_code
    cur.execute(sql)
    record = cur.fetchone()
    gid = record['globalid']

    conn.close()


    conn  = MySQLdb.connect(**config.db_asset)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)
    if op == 'delete':
        sql = "delete from ra_pool_fund where ra_pool = %d and ra_fund_id = %d" % (int(pool_id), int(gid))
        print sql
        cur.execute(sql)
    elif op == 'insert':
        sql = 'select ra_date from ra_pool_fund where ra_pool = %d' % (int)(pool_id)
        cur.execute(sql)
        dates = []
        records = cur.fetchall()
        for record in records:
            dates.append(record['ra_date'])
        dates = set(dates)

        for d in dates:
            sql = "replace into ra_pool_fund (ra_pool, ra_category, ra_date, ra_fund_id, ra_fund_code, ra_fund_type, ra_fund_level, ra_mask, created_at, updated_at) values ('%d', 0, '%s', '%d', '%s', 2, 1, 0, '%s', '%s')" % (int(pool_id), d, gid, fund_code, datetime.now(), datetime.now())
            print sql
            cur.execute(sql)

    conn.close()
