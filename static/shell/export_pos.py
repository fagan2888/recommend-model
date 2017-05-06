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
    "db":"asset_allocation",
    "charset": "utf8"
}


if __name__ == '__main__':


    conn  = MySQLdb.connect(**db_base)

    sql = 'select ra_date, ra_fund_code, sum(ra_fund_ratio) from ra_portfolio_pos where ra_portfolio_id = 80050201 group by ra_date, ra_fund_code'

    df = pd.read_sql(sql, conn, index_col = ['ra_date', 'ra_fund_code'])
    #df = df.groupby(by = df.index, group_keys = False).sum()
    #print df
    #print df.index
    df = df.unstack()
    df = df.fillna(0.0)
    print df

    df.to_csv('data/risk1_pos.csv')
