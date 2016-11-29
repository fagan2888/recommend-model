#coding=utf8


import MySQLdb
import pandas as pd


db_base = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"mofang",
    "charset": "utf8"
}


db_params = {
            "host": "rdsijnrreijnrre.mysql.rds.aliyuncs.com",
            "port": 3306,
            "user": "koudai",
            "passwd": "Mofang123",
            "db":"caihui",
            "charset": "utf8"
        }


if __name__ == '__main__':


    conn  = MySQLdb.connect(**db_base)
    sql   = 'select ra_code from ra_fund where ra_type = 1 and ra_mask = 0'
    cur = conn.cursor()
    cur.execute(sql)
    records = cur.fetchall()
    codes = []
    for record in records:
        codes.append(record[0])
    conn.close()


    conn  = MySQLdb.connect(**db_params)
    imploded_codes = ','.join([repr(code.encode('utf8')) for code in codes])
    #print imploded_codes
    sql = 'select SECODE, FSYMBOL, OUTSUBBEGDATE ,FDSNAME from TQ_FD_BASICINFO where FSYMBOL in (%s)' % imploded_codes
    df = pd.read_sql(sql, conn, index_col = 'FSYMBOL')
    df.to_csv('./data/all_stock_fund.csv', encoding='utf8')


    '''
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
    sql = 'select FDSNAME, MANAGERNAME, FSYMBOL, FOUNDDATE, SECURITYID, SECODE from TQ_FD_BASICINFO where ISVALID = 1 and SECURITYID in (' + imploded_sids + ')'
    df  = pd.read_sql(sql, conn, index_col = 'SECURITYID', parse_dates = ['FOUNDDATE'])
    #print df.loc['161211']
    df.reset_index(inplace = True)
    df = df.sort_values(by = ['FOUNDDATE'])
    df = df.groupby(['FSYMBOL']).last()
    df.set_index('SECURITYID')
    df.to_csv('all_stock_fund.csv', encoding='utf8')
    '''
