#coding=utf8


import pandas as pd
import numpy  as np
from datetime import datetime
import MySQLdb


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

    fund_df = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECURITYID', parse_dates = ['FOUNDDATE'])

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


    sql = 'select SECURITYID, INVTOTRTO, ENDDATE from TQ_FD_SHARESTAT where ISVALID = 1 and SECURITYID in (' + imploded_sids + ')'
    inv_df = pd.read_sql(sql, conn, ['ENDDATE', 'SECURITYID'], parse_dates = ['ENDDATE'])
    #inv_df = nav_df.groupby(level = (0, 1)).first()
    inv_df = inv_df.unstack()
    inv_df.columns = inv_df.columns.droplevel(0)
    dates = pd.date_range('2004-06-30',',2016-11-07')
    #print dates
    inv_df = inv_df.reindex(dates)
    inv_df.fillna(method = 'pad', inplace=True)
    #print inv_df

    codes = []
    for col in inv_df.columns:
       codes.append(sid_code[int(col)])
    inv_df.columns = codes
    inv_df.to_csv('fund_inv_ratio.csv')
    print inv_df
