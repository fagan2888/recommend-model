#coding=utf8


import MySQLdb
import pandas as pd
import numpy  as np
import datetime


db_params = {
            "host": "101.201.81.170",
            "port": 3306,
            "user": "wind",
            "passwd": "wind",
            "db":"caihui",
            "charset": "utf8"
        }


icodes = ['000300','000905','399314', '399316', '399372', '399373','399376','399377']


#所有股票的收盘价
def all_index_price():

    conn = MySQLdb.connect(**db_params)
    cur  = conn.cursor()

    sql = "select SECODE, SYMBOL from TQ_IX_BASICINFO where SYMBOL = '%s'"
    secode_symbol_dict = {}
    for icode in icodes:
        isql = sql % icode
        cur.execute(isql)
        record = cur.fetchone()
        secode = record[0].strip()
        symbol = record[1].strip()
        secode_symbol_dict[secode] = symbol
    cur.close()


    #print secode_symbol_dict
    secodes = secode_symbol_dict.keys()

    dfs = []
    for secode in secodes:
        sql = 'select TRADEDATE, TCLOSE from TQ_QT_INDEX where SECODE = %s order by TRADEDATE asc' % secode
        print sql
        df = pd.read_sql(sql, conn, index_col = 'TRADEDATE', parse_dates = ['TRADEDATE'])
        df.index.name = 'date'
        df.columns    = [secode_symbol_dict[secode]]
        df.replace(0.00, np.nan, inplace = True)
        dfs.append(df)
        #print secode_symbol_dict[secode], 'done'
    stock_df = pd.concat(dfs, axis = 1)

    stock_df.to_csv('index_price.csv')


if __name__ == '__main__':
    all_index_price()
