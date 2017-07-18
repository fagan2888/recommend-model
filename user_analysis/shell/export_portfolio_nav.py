#coding=utf8


import MySQLdb
import pandas as pd


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

    sql = 'select ra_date as date, ra_nav as nav from ra_portfolio_nav where ra_portfolio_id = 80071810 and ra_type = 9'
    df = pd.read_sql(sql, conn, index_col = ['date'])
    print df
    df.to_csv('./tmp/portfolio_nav.csv')
