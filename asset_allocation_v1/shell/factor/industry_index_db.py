#coding=utf8



import MySQLdb
import pandas as pd
import numpy  as np
import datetime



db_params = {
            "host": "rdsijnrreijnrre.mysql.rds.aliyuncs.com",
            "port": 3306,
            "user": "koudai",
            "passwd": "Mofang123",
            "db":"caihui",
            "charset": "utf8"
        }


shenwan_industry_codes = [801010, 801020, 801030, 801040, 801050, 801080, 801110, 801120, 801130, 801140, 801150, 801160, 801170, 801180, 801200, 801210, 801230, 801710, 801720,801730, 801740, 801750, 801760, 801770, 801780, 801790, 801880, 801890]


if __name__ == '__main__':


    conn = MySQLdb.connect(**db_params)


    imploded_codes = ','.join([repr(code) for code in shenwan_industry_codes])
    #print imploded_codes
    sql = 'select SECODE, INDEXNAME, SYMBOL from tq_ix_basicinfo where SYMBOL in (' + imploded_codes + ')'
    #print sql
    df = pd.read_sql(sql, conn)

    secode_symbol = {}
    for i in range(0, len(df)):
        secode = df.iloc[i]['SECODE']
        symbol = df.iloc[i]['SYMBOL']
        #print secode, symbol
        secode_symbol.setdefault(secode, symbol)

    imploded_secodes = ','.join([repr(secode.encode('utf8')) for secode in df['SECODE'].values])

    sql = 'select TRADEDATE, SECODE, TCLOSE from TQ_QT_INDEX where SECODE in (' + imploded_secodes + ')'

    #print sql
    df = pd.read_sql(sql, conn, index_col = ['TRADEDATE', 'SECODE'], parse_dates = ['TRADEDATE'])
    df = df.unstack(level = 1)
    #print df
    df.columns = df.columns.get_level_values(1)
    cols = []
    for col in df.columns:
        symbol = secode_symbol[col]
        cols.append(str(symbol) + '.index')
    df.columns = cols
    #print df
    print df
    df.to_csv('industry_index.csv')
