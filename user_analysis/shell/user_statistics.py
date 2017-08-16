#coding=utf8

import MySQLdb
import pandas as pd


portfolio_statistics = {
    "host": "rdsf4ji381o0nt6n2954.mysql.rds.aliyuncs.com",
    "port": 3306,
    "user": "jiaoyang",
    "passwd": "wgOdGq9SWruwATrVWGwi",
    "db":"portfolio_statistics",
    "charset": "utf8"
}


if __name__ == '__main__':


    conn  = MySQLdb.connect(**portfolio_statistics)
    conn.autocommit(True)

    sql = 'select * from user_statistics'

    df = pd.read_sql(sql, conn, index_col = ['us_date'], parse_dates = ['us_date'])
    print df

    df.to_csv('user.csv')
