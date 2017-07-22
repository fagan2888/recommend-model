#coding=utf8


import sys
sys.path.append('shell')
import MySQLdb
import config
import pandas as pd



if __name__ == '__main__':


    conn  = MySQLdb.connect(**config.db_base)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)


    sql = 'select globalid, ra_code, ra_mask from ra_fund'
    df = pd.read_sql(sql, conn, index_col = ['globalid'])

    df = pd.read_csv()
