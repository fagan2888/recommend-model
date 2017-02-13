#coding=utf8


import pandas as pd
import numpy  as np
from datetime import datetime
import MySQLdb


db_params = {
            "host": "rdsijnrreijnrre.mysql.rds.aliyuncs.com",
            "port": 3306,
            "user": "koudai",
            "passwd": "Mofang123",
            "db":"caihui",
            "charset": "utf8"
        }


if __name__ == '__main__':


    fund_df = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECODE')


    secodes = fund_df.index.values
    imploded_secodes = ','.join([repr(seid) for seid in secodes])

    conn  = MySQLdb.connect(**db_params)

    sql = 'select ENDDATE, SECODE, SKCODE ,NAVRTO from TQ_FD_SKDETAIL where SECODE in (' + imploded_secodes + ')'
    df = pd.read_sql(sql, conn, index_col = ['ENDDATE'])
    df = df.fillna(0.0)
    df['NAVRTO'] = df['NAVRTO'] / 100
    #print df

    conn.close()

    df.to_csv('./data/tq_fd_skdetail.csv')
