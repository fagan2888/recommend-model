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


if __name__ == '__main__':


    conn  = MySQLdb.connect(**db_base)
    sql   = 'select SECURITYID, L1NAME, L1CODES, L2NAME, L2CODES, L3NAME, L3CODES from TQ_FD_TYPECLASS'
    df    = pd.read_sql(sql, conn)
    df.to_csv('fund_type.csv', encoding = 'utf8')

    codes = set()
    for code in df.L2CODES.values:
       if code.find('2001') == 0:
            codes.add(code)
    codes.remove('200103')
    codes.add('200201')
    codes.add('200202')



    securityids = set()
    for code in codes:
        tmp_df = df[df['L2CODES'] == code]['SECURITYID']
        for sid in tmp_df.values:
            securityids.add(sid)


    imploded_sids = ','.join([repr(sid.encode('utf8')) for sid in securityids])
    sql = 'select FDSNAME, MANAGERNAME, FSYMBOL, FOUNDDATE, SECURITYID from TQ_FD_BASICINFO where ISVALID = 1 and SECURITYID in (' + imploded_sids + ')'
    df  = pd.read_sql(sql, conn, index_col = 'SECURITYID', parse_dates = ['FOUNDDATE'])
    #print df.loc['161211']
    df.reset_index(inplace = True)
    df = df.sort_values(by = ['FOUNDDATE'])
    df = df.groupby(['FSYMBOL']).last()
    df.set_index('SECURITYID')
    df.to_csv('all_stock_fund.csv', encoding='utf8')
