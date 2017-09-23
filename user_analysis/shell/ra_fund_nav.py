#coding=utf8


import MySQLdb
import pandas as pd
import numpy as np
import datetime


mofang = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"mofang",
    "charset": "utf8"
}


if __name__ == '__main__':

    conn  = MySQLdb.connect(**mofang)
    conn.autocommit(True)

    sql = 'select td_date from trade_dates'
    df = pd.read_sql(sql, conn)
    dates = []
    for d in df.values:
        dates.append(d[0].strftime('%Y-%m-%d'))

    sql = 'select ra_code as code from ra_fund where ra_type = 1'
    df = pd.read_sql(sql, conn)
    codes = ','.join(["'" + code[0] + "'" for code in df.values])

    sql = 'select ra_code as code, ra_date as date, ra_nav_adjusted as nav from ra_fund_nav where ra_code in (' + codes + ')'

    df = pd.read_sql(sql, conn, index_col = ['date', 'code'])
    df = df.unstack()
    df.index = df.index.astype(str)

    dates = set(dates) & set(df.index)
    dates = list(dates)
    dates.sort()
    df = df.loc[dates]
    #df = df[~df.index.duplicated(keep = 'first')]
    #df = df.unstack()

    df.columns = df.columns.get_level_values(1)
    df.index.name = 'date'
    print df
    df.to_csv('./data/fund_nav.csv')
