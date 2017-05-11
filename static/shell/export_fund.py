#coding=utf8


import pandas as pd
import numpy as np
import scipy.optimize
import MySQLdb


db_base = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"mofang",
    "charset": "utf8"
}


if __name__ == '__main__':

    conn  = MySQLdb.connect(**db_base)

    sql = 'select ra_code, ra_date, ra_nav_adjusted from ra_fund_nav'

    df = pd.read_sql(sql, conn, index_col = ['ra_date', 'ra_code'])
    df = df.unstack()
    #df = df.dropna()
    df = df.fillna(method = 'pad')
    #print df
    df.to_csv('fund_value.csv')
