#coding=utf8


import MySQLdb
import pandas as pd
import numpy  as np
import datetime


db_params = {
            "host": "101.201.81.170",
            "port": 3306,
            "user": "wind",
            "passwd": "wind",
            "db":"caihui_test",
            "charset": "utf8"
        }


#所有股票的收盘价
def all_stock_code():

    conn = MySQLdb.connect(**db_params)

    sql = 'select SECURITYID, SYMBOL, SECODE from TQ_SK_BASICINFO'

    df = pd.read_sql(sql, conn, index_col = 'SECURITYID')
    print df
    df.to_csv('./data/stock_code.csv')
    return df


if __name__ == '__main__':
    all_stock_code()

