#coding=utf8


import MySQLdb
import pandas as pd
import datetime


asset_allocation = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"mofang",
    "charset": "utf8"
}


if __name__ == '__main__':


    conn  = MySQLdb.connect(**asset_allocation)
    conn.autocommit(True)

    indexs = [120000001, 120000002, 120000013, 120000014,120000015]
    codes = ','.join(["'" + str(code) + "'" for code in indexs])

    sql = 'select ra_index_code as code, ra_date as date, ra_nav as nav from ra_index_nav where ra_index_id in (' + codes + ')'
    df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])
    df = df.reset_index()
    df = df.set_index(['date', 'code'])
    df = df.unstack()
    df.columns = df.columns.get_level_values(1)
    df.to_csv('./data/index.csv')
    #turnover = abs(df.diff().fillna(0.0)).sum().sum() / len(df)
    #print 'risk' + str(i), len(df) / 12.0, turnover
