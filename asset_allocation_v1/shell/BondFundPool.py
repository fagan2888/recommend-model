# -*- coding: UTF-8 -*-
import MySQLdb
import pandas as pd
import os

db_base = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"caihui",
    "charset": "utf8"
}

conn  = MySQLdb.connect(**db_base)

def load_save_fund_types():
    sql   = 'select SECURITYID, L1NAME, L1CODES, L2NAME, L2CODES, L3NAME, L3CODES from TQ_FD_TYPECLASS'
    df    = pd.read_sql(sql, conn)
    df.to_csv('../tmp/fund_types.csv', encoding='utf8')

def load_bond_funds(types):
    """
    加载types里指定分类代码的债券基金
    :param types: 债券基金分类代码
    :return: 返回相应代码基金,以dataframe格式返回
    """
    # load_save_fund_types()
    ftypes_df = pd.read_csv('../tmp/fund_types.csv', index_col='ids')
    securityids = set()
    for code in types:
        tmp_df = ftypes_df[ftypes_df['L2CODES'] == code]['SECURITYID']
        for sid in tmp_df.values:
            securityids.add(sid)
    #print securityids
    #for sid in securityids:
    #    print type(sid)
    #    os._exit(0) 
    imploded_sids = ','.join([repr(sid) for sid in securityids])
    sql = 'select FDSNAME, MANAGERNAME, FSYMBOL, FOUNDDATE, SECURITYID from TQ_FD_BASICINFO where ISVALID = 1 and SECURITYID in (' + imploded_sids + ')'
    df  = pd.read_sql(sql, conn, index_col = 'SECURITYID', parse_dates = ['FOUNDDATE'])
    #print df.loc['161211']
    #df.reset_index(inplace = True)
    #df = df.sort_values(by = ['FOUNDDATE'])
    #df = df.groupby(['FSYMBOL']).last()
    #df.set_index('SECURITYID')
    df.to_csv("../tmp/bondfunds.csv", encoding="utf8")
    # delta = datetime.datetime(2016, 1, 1) - datetime.datetime(2015, 1, 1)
    # delta.days()
    #print df[:-1]

if __name__ == "__main__":
    load_bond_funds([200204, 200301, 200302, 200306])
