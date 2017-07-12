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

    sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 800001 and on_type = 8'
    df = pd.read_sql(sql, conn, index_col = ['date'])
    print df
    df.to_csv('./tmp/online_nav.csv')
