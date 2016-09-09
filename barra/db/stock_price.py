#coding=utf8


import MySQLdb
import pandas as pd
import numpy  as np


db_params = {
            "host": "101.201.81.170",
            "port": 3306,
            "user": "wind",
            "passwd": "wind",
            "db":"caihui_test",
            "charset": "utf8"
        }


#所有股票的收盘价
def all_stock_price():

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

    #print secode_symbol_dict
    secodes = secode_symbol_dict.keys()

    dfs = []
    for secode in secodes:
        sql = 'select TRADEDATE, TCLOSE from TQ_QT_SKDAILYPRICE where SECODE = %s order by TRADEDATE asc' % secode
        df = pd.read_sql(sql, conn, index_col = 'TRADEDATE', parse_dates = ['TRADEDATE'])
        df.index.name = 'date'
        df.columns    = [secode_symbol_dict[secode]]
        df.replace(0.00, np.nan, inplace = True)
        dfs.append(df)
        print secode_symbol_dict[secode], 'done'
    stock_df = pd.concat(dfs, axis = 1)

    stock_df.to_csv('stock_price.csv')


#所有股票的收盘价
def all_stock_price_update():

    stock_df = pd.read_csv('stock_price.csv', index_col = 'date', parse_dates = ['date'])

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

    #print secode_symbol_dict
    secodes = secode_symbol_dict.keys()

    dfs = []
    for secode in secodes:
        sql = 'select TRADEDATE, TCLOSE from TQ_QT_SKDAILYPRICE where SECODE = %s order by TRADEDATE asc limit 0, 10' % secode
        df = pd.read_sql(sql, conn, index_col = 'TRADEDATE', parse_dates = ['TRADEDATE'])
        df.index.name = 'date'
        df.columns    = [secode_symbol_dict[secode]]
        df.replace(0.00, np.nan, inplace = True)
        dfs.append(df)
        print secode_symbol_dict[secode], 'done'

    df = pd.concat(dfs, axis = 1)

    #stock_df = pd.
    #stock_df.to_csv('stock_price.csv')
    print df


if __name__ == '__main__':
    all_stock_price_update()
