#coding=utf8


import MySQLdb
import pandas as pd
import datetime


asset_allocation = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"asset_allocation",
    "charset": "utf8"
}


if __name__ == '__main__':

    conn  = MySQLdb.connect(**asset_allocation)
    conn.autocommit(True)

    for i in range(0, 10):
        sql = 'select ra_date as date, ra_fund_code as code, ra_fund_ratio as ratio from ra_portfolio_pos where ra_portfolio_id = 8008140%d' % i
        df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])
        df = df[df.index >= '2012-09-25']
        df = df.reset_index()
        df = df.groupby(['date', 'code']).sum()
        df = df.unstack().fillna(0.0)
        #turnover = abs(df.diff().fillna(0.0)).sum().sum() / len(df)
        print abs(df.diff().fillna(0.0)).sum(axis = 1)
        #print 'risk' + str(i), len(df) / 12.0, turnover
