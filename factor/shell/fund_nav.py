#coding=utf8


import MySQLdb
import pandas as pd
import numpy  as np


db_params = {
            "host": "127.0.0.1",
            "port": 3306,
            "user": "root",
            "passwd": "Mofang123",
            "db":"mofang",
            "charset": "utf8"
        }


if __name__ == '__main__':

    conn = MySQLdb.connect(**db_params)

    codes = []
    sql  = 'select distinct(ra_code) from ra_fund where ra_type = 1'
    cur  = conn.cursor()
    cur.execute(sql)

    for record in cur.fetchall():
        codes.append(record[0])
    cur.close()


    dfs = []
    for code in codes:
        sql = "select ra_date, ra_nav_adjusted from ra_fund_nav where ra_code = '%s'" % code
        df = pd.read_sql(sql, conn, index_col = 'ra_date', parse_dates = ['ra_date'])
        df.index.name = 'date'
        df.columns    = [code]
        df.replace(0.00, np.nan, inplace = True)
        dfs.append(df)
        print code, 'done'
    stock_df = pd.concat(dfs, axis = 1)


    print stock_df
    stock_df.to_csv('fund_value.csv')




    codes = []
    sql  = 'select distinct(ra_code) from ra_index'
    cur  = conn.cursor()
    cur.execute(sql)

    for record in cur.fetchall():
        codes.append(record[0])
    cur.close()


    dfs = []
    for code in codes:
        sql = "select ra_date, ra_nav from ra_index_nav where ra_index_code = '%s'" % code
        df = pd.read_sql(sql, conn, index_col = 'ra_date', parse_dates = ['ra_date'])
        df.index.name = 'date'
        df.columns    = [code]
        df.replace(0.00, np.nan, inplace = True)
        dfs.append(df)
        print code, 'done'
    stock_df = pd.concat(dfs, axis = 1)


    print stock_df
    stock_df.to_csv('index_value.csv')

    conn.close()
