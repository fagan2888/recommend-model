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


    fund_df = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECURITYID', parse_dates = ['FOUNDDATE'])

    conn  = MySQLdb.connect(**db_base)

    codes = []
    for code in fund_df.index:
        codes.append(code)

    #imploded_sids = ','.join([repr(sid) for sid in codes])

    dfs = []
    base_sql = 'select ENDDATE, SECODE, SKCODE, SKNAME, HOLDMKTCAP, HOLDAMT, NAVRTO, ACCSTKRTO, ACCCIRCRTO, SECURITYID from TQ_FD_SKDETAIL where SECURITYID = %d'
    for code in codes:
        #print code
        sql = base_sql % code
        #print sql
        df = pd.read_sql(sql, conn, index_col = 'SECURITYID')
        print df
        #print df
        dfs.append(df)

    df = pd.concat(dfs)
    print df
    df.to_csv('tq_fd_skdetail')
    #print sql
    #df = pd.read_sql(sql, conn, index_col = 'SECURITYID')
    #print df
    #print imploded_sids
