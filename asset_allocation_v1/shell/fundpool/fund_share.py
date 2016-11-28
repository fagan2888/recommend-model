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

    conn  = MySQLdb.connect(**db_params)

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

    sql = 'select SECURITYID, ENDDATE, TOTALSHARE from TQ_FD_FSHARE where ISVALID = 1 and SECURITYID in (' + imploded_sids + ')'

    share_df = pd.read_sql(sql, conn, index_col = ['ENDDATE', 'SECURITYID'], parse_dates = ['ENDDATE'])
    share_df = share_df.unstack()
    share_df.columns = share_df.columns.droplevel(0)
    codes = []
    for col in share_df.columns:
       codes.append(sid_code[int(col)])

    share_df.fillna(method = 'pad', inplace=True)
    share_df.columns = codes
    share_df.index.name = 'date'
    print share_df
    share_df.to_csv('share.csv')

    sql = 'select SECURITYID, UNITNAV, NAVDATE from TQ_QT_FDNAV where ISVALID = 1 and SECURITYID in (' + imploded_sids + ')'
    nav_df = pd.read_sql(sql, conn, ['NAVDATE', 'SECURITYID'], parse_dates = ['NAVDATE'])
    nav_df = nav_df.groupby(level = (0, 1)).first()
    nav_df = nav_df.unstack()
    nav_df.columns = nav_df.columns.droplevel(0)
    #print nav_df
    codes = []
    for col in nav_df.columns:
       codes.append(sid_code[int(col)])
    nav_df.columns = codes
    nav_df.fillna(method = 'pad')
    nav_df.index.name = 'date'
    print nav_df
    nav_df.to_csv('nav.csv')


    codes = set(share_df.columns.values) & set(nav_df.columns.values)
    codes = list(codes)
    print codes
    nav_df = nav_df[codes]
    share_df = share_df[codes]

    dates = set(nav_df.index.values) | set(share_df.index.values)
    dates = list(dates)
    dates.sort()
    nav_df = nav_df.reindex(dates)
    share_df = share_df.reindex(dates)
    nav_df.fillna(method = 'pad', inplace = True)
    share_df.fillna(method = 'pad', inplace = True)

    #share_df = share_df.iloc[:,0:10]
    #nav_df = nav_df.iloc[:,0:10]
    nav_df.to_csv('nav.csv')
    share_df.to_csv('share.csv')


    fund_size_df = share_df * nav_df
    dates = pd.date_range('2010-01-01', '2016-11-09')
    fund_size_df = fund_size_df.reindex(dates)
    fund_size_df.fillna(method = 'pad', inplace = True)
    print fund_size_df
    fund_size_df.to_csv('./fund_size.csv')
