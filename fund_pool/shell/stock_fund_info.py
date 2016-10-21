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
    "host": "101.201.81.170",
    "port": 3306,
    "user": "wind",
    "passwd": "wind",
    "db":"caihui_test",
    "charset": "utf8"
}
'''


if __name__ == '__main__':

    conn  = MySQLdb.connect(**db_base)

    df = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECURITYID')
    codes = []
    for code in df.index:
        codes.append(code)


    '''
    imploded_sids = ','.join([repr(sid.encode('utf8')) for sid in codes])
    #sql = 'select FDSNAME, FDNATURE, INVESTSTYLE, FOUNDDATE, SECURITYID, FSYMBOL, MANAGERNAME from TQ_FD_BASICINFO where SECURITYID in (' + imploded_sids + ')'
    sql = 'select * from TQ_FD_BASICINFO where SECURITYID in (' + imploded_sids + ')'
    fund_info_df = pd.read_sql(sql, conn, index_col = 'SECURITYID')
    print fund_info_df
    #fund_info_df.to_csv('tq_fd_basicinfo', encoding = 'utf8')


    #sql             = 'select MANAGERCODE, MANAGERNAME, NAVRTO, BENCHMARKRTO, TENUREYIELD TENUREYIELDYR, ALLRANK, ALLAVGYIELD, CLASSRANK, RANKTYPE, RTYPEAVGYIELD , SECURITYID from TQ_FD_MGPERFORMANCE where SECURITYID in (' + imploded_sids + ')'
    sql             = 'select * from TQ_FD_MGPERFORMANCE where SECURITYID in (' + imploded_sids + ')'
    mgperformance_df= pd.read_sql(sql, conn, index_col = 'SECURITYID')
    print mgperformance_df
    #mgperformance_df.to_csv('tq_fd_mgperformance', encoding = 'utf8')


    #mgperformance_df.to_csv('./tmp/manager_performance.csv', encoding='utf8')
    #print mgperformance_df

    #sql = 'select PSCODE, PSNAME, GENDER, BIRTH, DEGREE, TOTYEARS, CURFCOUNT, AVGTENURE, TOTCOUNT from TQ_FD_MANAGERSTA'
    sql = 'select * from TQ_FD_MANAGERSTA'
    managersta_df = pd.read_sql(sql, conn)
    print managersta_df
    #managersta_df.to_csv('tq_fd_managersta' ,encoding = 'utf8')
    '''

    base_sql = 'select * from TQ_FD_DERIVEDN where SECURITYID = %d'
    for code in codes:
        sql = base_sql % code
        print sql
        derivedn_df  = pd.read_sql(sql, conn)
        print derivedn_df

    #derivedn_df.to_csv('tq_fd_derivedn.csv', encoding = 'utf8')
    #mgperformance_df= mgperformance_df.loc[fund_info_df.index]

    #df = pd.concat([fund_info_df, mgperformance_df])
    #print df
    #print df.head()
