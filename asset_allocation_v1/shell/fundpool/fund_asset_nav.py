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



    '''
    conn  = MySQLdb.connect(**db_base)

    df = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECURITYID')
    codes = []
    for code in df.index:
        codes.append(code)


    imploded_sids = ','.join([repr(sid) for sid in codes])

    sql = 'select ENTRYDATE, ASSETNAV, SECODE, SECURITYID from TQ_QT_FDNAV where SECURITYID in (' + imploded_sids + ')'

    df = pd.read_sql(sql, conn, index_col = 'SECURITYID')

    print df

    df.to_csv('assetnav.csv')
    '''

    '''
    df = pd.read_csv('./assetnav.csv', index_col = 'SECODE')
    df.reset_index(inplace = True)
    df.set_index(['ENTRYDATE', 'SECODE'], inplace = True)
    #print df
    df.to_csv('assetnav.csv')
    '''

    conn  = MySQLdb.connect(**db_base)

    df = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECURITYID')
    codes = []
    for code in df.index:
        codes.append(code)
    imploded_sids = ','.join([repr(sid) for sid in codes])


    #base_sql = 'select SECODE, ENTRYDATE, SECURITYID, INSTINVESHARE'
    #df = pd.read_sql(
    print imploded_sids
