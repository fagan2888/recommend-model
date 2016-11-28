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

    df = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECURITYID')
    codes = []
    for code in df.index:
        codes.append(code)
    imploded_sids = ','.join([repr(sid) for sid in codes])


    sql = 'select SECURITYID, TOTALSHARE, INSTINVESHARE, DECLAREDATE from TQ_FD_FSHARE WHERE SECURITYID in (' + imploded_sids + ')'

    df = pd.read_sql(sql, conn, index_col = 'SECURITYID')

    print df
