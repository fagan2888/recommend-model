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



if __name__ == '__main__':


    conn = MySQLdb.connect(**db_params)
    cur  = conn.cursor()


    sql = 'select SECODE, SYMBOL from TQ_SK_BASICINFO'
    cur.execute(sql)

    secode_symbol_dict = {}
    for record in cur.fetchall():
        secode = record[0]
        symbol = record[1]
        if symbol.find('00') == 0 or symbol.find('60') == 0 or symbol.find('30') == 0:
            #print symbol , secode
            secode_symbol_dict[secode] = symbol
    cur.close()

    #imploded_secodes = ','.join([repr(seid) for seid in secodes])
    imploded_symbols = ','.join([repr(seid.encode('utf8')) for seid in secode_symbol_dict.values()])

    sql = 'select SYMBOL, SWLEVEL1CODE from tq_sk_basicinfo where SYMBOL in (' + imploded_symbols + ')'

    df = pd.read_sql(sql, conn, index_col = ['SYMBOL'])
    print df
