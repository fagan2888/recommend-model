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

    sql = 'select on_date, on_asset_id, on_ratio from on_online_markowitz where on_online_id = 800010'

    df = pd.read_sql(sql, conn, index_col = ['on_date', 'on_asset_id'])
    df = df.unstack()
    df = df.fillna(0.0)
    print df

    df.to_csv('data/pos.csv')
