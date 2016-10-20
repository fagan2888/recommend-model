#coding=utf8


import MySQLdb
import pandas as pd


db_base = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"caihui",
    "charset": "utf8"
}


'''
db_base = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"caihui",
    "charset": "utf8"
}
'''


if __name__ == '__main__':

    conn  = MySQLdb.connect(**db_base)

    df = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECURITYID')
    codes = []
    for code in df.index:
        codes.append('%06d' % code)

    imploded_sids = ','.join([repr(sid.encode('utf8')) for sid in codes])
    sql             = 'select MANAGERCODE, MANAGERNAME, NAVRTO, BENCHMARKRTO, TENUREYIELD TENUREYIELDYR, ALLRANK, ALLAVGYIELD, CLASSRANK, RANKTYPE, RTYPEAVGYIELD , SECURITYID from TQ_FD_MGPERFORMANCE where SECURITYID in (' + imploded_sids + ')'
    mgperformance_df= pd.read_sql(sql, conn, index_col = 'SECURITYID')


    mgperformance_df.to_csv('./tmp/manager_performance.csv', encoding='utf8')
    #print mgperformance_df

    #sql = 'select PSCODE, PSNAME, GENDER, BIRTH, DEGREE, TOTYEARS, URFCOUNT, AVGTENURE, TOTCOUNT from TQ_FD_MANAGERSTA'
    #managersta_df = pd.read_sql(sql, conn)
    #print managersta_df

    #mgperformance_df= mgperformance_df.loc[fund_info_df.index]

    #df = pd.concat([fund_info_df, mgperformance_df])
    #print df
    #print df.head()
