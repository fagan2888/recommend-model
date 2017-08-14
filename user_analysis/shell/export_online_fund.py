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
        sql = 'select on_date as date, on_fund_code as code, on_fund_ratio as ratio from on_online_fund where on_online_id = 80000%d' % i
        df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])
        df = df[df.index >= '2016-07-01']
        df = df[df.index <= '2017-07-31']
        df = df.reset_index()
        df = df.set_index(['date', 'code'])
        df = df.unstack().fillna(0.0)
        turnover = abs(df.diff().fillna(0.0)).sum().sum() / len(df)
        print 'risk' + str(i), len(df) / 12.0, turnover
