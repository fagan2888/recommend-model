#coding=utf8


import pandas as pd
import numpy as np
import MySQLdb
from datetime import datetime


db_base = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"mofang",
    "charset": "utf8"
}


if __name__ == '__main__':


    conn  = MySQLdb.connect(**db_base)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)


    index_base_sql = "replace into index_info (ii_globalid, ii_index_code, ii_caihui_code, ii_name, ii_base_date, ii_begin_date, ii_announce_date, created_at, updated_at) values (%d, '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')"
    df = pd.read_csv('./data/factor_index.csv', parse_dates = ['date'], index_col = ['date'])
    begin_date = df.index[0]


    factor_names = df.columns
    fid_dict = {'ln_capital_L':120100010,'ln_capital_S':120100011,'BP_L':120100020,'BP_S':120100021,'std_3m_L':120100030,'std_3m_S':120100031, 'tradevolumn_3m_L':120100040, 'tradevolumn_3m_S':120100041, 'holder_avgpct_L':120100050, 'holder_avgpct_S':120100051}
    for fname in df.columns:
        fid = fid_dict[fname]
        sql = index_base_sql % (fid, fname, '', fname, begin_date, begin_date, begin_date, datetime.now(), datetime.now())
        cur.execute(sql)


    index_nav_base_sql = "replace into index_value (iv_index_id, iv_index_code, iv_time, iv_value, iv_open, iv_high, iv_low, created_at, updated_at) values ('%d', '%s', '%s', '%f', '%f', '%f', '%f', '%s', '%s')"
    df = pd.read_csv('./data/factor_fund_pool.csv', parse_dates = ['date'], index_col = ['date'])
    for fname in df.columns:
        fid = fid_dict[fname]
        ser = df[fname]
        for d in ser.index:
            sql = index_nav_base_sql % (fid, fname, d, ser.loc[d], 0, 0, 0, datetime.now(), datetime.now())
            print sql
            cur.execute(sql)

    conn.close()
