#coding=utf8


import MySQLdb
import pandas as pd


db_base = {
    "host": "101.201.81.170",
    "port": 3306,
    "user": "wind",
    "passwd": "wind",
    "db":"caihui_test",
    "charset": "utf8"
}


if __name__ == '__main__':

    conn  = MySQLdb.connect(**db_base)
    fund_df = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECURITYID')

    base_sql = 'select ENDDATE, REPAIRUNITNAV, SECODE, REPAIRUNITNAVG from TQ_FD_DERIVEDN where SECURITYID = %d'

    for sid in fund_df.index:
        sql = base_sql % sid
        nav_df = pd.read_sql(sql, conn)
        print nav_df
