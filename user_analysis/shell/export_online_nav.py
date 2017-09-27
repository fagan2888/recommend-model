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

    sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 800000 and on_type = 9'
    online_df = pd.read_sql(sql, conn, index_col = ['date'])
    online_df.to_csv('./tmp/online_nav.csv')


    df = pd.read_csv('data/index.csv', parse_dates = ['date'], index_col = ['date'])
    df = df[['000300.SH']]
    df = pd.concat([online_df, df], axis = 1, join_axes = [online_df.index])
    df = df.dropna()
    df = df / df.iloc[0]
    df.to_csv('./data/nav.csv')
    print df
    '''
    dfs = []
    for i in range(0, 10):
        sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 80000%d and on_type = 9' % i
        df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])
        df.columns = ['risk_' + str(i)]
        dfs.append(df)

    df = pd.concat(dfs, axis = 1)
    df = df[df.index >= '2016-07-01']
    df = df[df.index <= '2017-07-31']
    df = df / df.iloc[0]
    print df
    df.to_csv('./tmp/online_nav.csv')
    '''
