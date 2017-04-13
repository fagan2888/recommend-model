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

    sql = 'select ra_date, ra_alloc_id, ra_nav from risk_asset_allocation_nav where ra_type = 9'

    df = pd.read_sql(sql, conn, index_col = ['ra_date', 'ra_alloc_id'])
    df = df.unstack()
    df = df.dropna()
    print df

    df.to_csv('data/risk_asset_allocation.csv')

