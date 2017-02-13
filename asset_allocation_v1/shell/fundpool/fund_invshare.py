#coding=utf8


import pandas as pd
import numpy  as np
from datetime import datetime
import MySQLdb

db_base = {
            "host": "rdsijnrreijnrre.mysql.rds.aliyuncs.com",
            "port": 3306,
            "user": "koudai",
            "passwd": "Mofang123",
            "db":"caihui",
            "charset": "utf8"
        }

if __name__ == '__main__':

    conn  = MySQLdb.connect(**db_base)

    fund_df = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECODE', parse_dates = ['OUTSUBBEGDATE'])

    codes = []
    for code in fund_df.index:
        codes.append(code)

    imploded_sids = ','.join([repr(sid) for sid in codes])
    #print imploded_sids

    sid_code = {}
    for securityid in fund_df.index:
        code = fund_df.loc[securityid, 'FSYMBOL']
        code = '%06d' % (int)(code)
        sid_code[securityid] = code


    sql = 'select SECODE, INVTOTRTO, ENDDATE from TQ_FD_SHARESTAT where ISVALID = 1 and SECODE in (' + imploded_sids + ')'
    inv_df = pd.read_sql(sql, conn, ['ENDDATE', 'SECODE'], parse_dates = ['ENDDATE'])
    #inv_df = nav_df.groupby(level = (0, 1)).first()
    inv_df = inv_df.unstack()
    inv_df.columns = inv_df.columns.droplevel(0)
    dates = pd.date_range('2004-06-30',',2017-02-09')
    #print dates
    inv_df = inv_df.reindex(dates)
    inv_df.fillna(method = 'pad', inplace=True)
    #print inv_df

    codes = []
    for col in inv_df.columns:
       codes.append(sid_code[int(col)])
    inv_df.columns = codes
    inv_df.index.name = 'date'
    inv_df.to_csv('./data/fund_inv_ratio.csv')
    print inv_df
