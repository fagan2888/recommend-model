#coding=utf8


import sys
sys.path.append('shell')
import config
import MySQLdb
import pandas as pd


if __name__ == '__main__':

    conn  = MySQLdb.connect(**config.db_asset)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)


    sql = 'select vw_view_id, vw_date, vw_inc from vw_view_inc'

    df = pd.read_sql(sql, conn, index_col = ['vw_date', 'vw_view_id'], parse_dates = ['vw_date'])
    df = df.unstack()

    df = df.to_csv('./data/view_format.csv')
