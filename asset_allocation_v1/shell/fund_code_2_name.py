#coding=utf8


import sys
sys.path.append('shell')
import LabelAsset
import pandas as pd
import DFUtil
import DBData
import numpy as np
import MySQLdb


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

    stock_fund_df = pd.read_csv('./tmp/stock_fund.csv', index_col = 'date', parse_dates = ['date'])
    #print stock_fund_df
    for date in stock_fund_df.index:
        codes = set()
        for col in stock_fund_df.columns:
            #print type(stock_fund_df.loc[date, col])
            v = stock_fund_df.loc[date, col]
            if type(v) is float:
                continue
            v = eval(stock_fund_df.loc[date, col])
            for code in v:
                codes.add(code)
        for code in codes:
            sql = 'select ra_code, ra_name from ra_fund where ra_code = %s' % code
            cur.execute(sql)
            record = cur.fetchone()
            print date, ',' ,record['ra_code'], ',' ,record['ra_name'].encode('utf8')

