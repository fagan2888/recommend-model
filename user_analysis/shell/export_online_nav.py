#coding=utf8


import MySQLdb
import pandas as pd


asset_allocation = {
    "host": "rdsf4ji381o0nt6n2954.mysql.rds.aliyuncs.com",
    "port": 3306,
    "user": "jiaoyang",
    "passwd": "wgOdGq9SWruwATrVWGwi",
    "db":"asset_allocation",
    "charset": "utf8"
}


if __name__ == '__main__':

    conn  = MySQLdb.connect(**asset_allocation)
    conn.autocommit(True)

    sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 800000 and on_type = 8'
    online_df = pd.read_sql(sql, conn, index_col = ['date'])
    online_df.to_csv('./tmp/online_nav.csv')


    df = pd.read_csv('data/000001.SH.csv', parse_dates = ['date'], index_col = ['date'])
    df = pd.concat([online_df, df], axis = 1, join_axes = [df.index])
    df = df.dropna()
    df.to_csv('./data/nav.csv')
