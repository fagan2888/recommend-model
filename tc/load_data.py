# -*- coding: UTF-8 -*-
import MySQLdb
import pandas as pd
import numpy as np
import os
import datetime
db_base = {
        "host": "rdsijnrreijnrre.mysql.rds.aliyuncs.com",
        "port": 3306,
        "user": "koudai",
        "passwd": "Mofang123",
        "db": "caihui",
        "charset": "utf8"
}
conn  = MySQLdb.connect(**db_base)
def load_index(symbol, stime, etime):
    cursor = conn.cursor()
    sql   = 'select SECODE from TQ_IX_BASICINFO where SYMBOL = "%s" ' % (symbol)
    cursor.execute(sql)
    secode = cursor.fetchone()[0]
    sql_index = 'select TRADEDATE AS date, LCLOSE AS close_pre, TCLOSE AS close, TOPEN AS open, THIGH AS high, TLOW AS low, VOL AS volume, AMOUNT AS amount, `CHANGE` AS returns, PCHG AS ratio FROM TQ_QT_INDEX where ISVALID = 1 and TRADEDATE >= ' + stime + ' and TRADEDATE <= ' + etime + ' and SECODE = ' + secode + ' ORDER BY TRADEDATE ASC'
    index_df = pd.read_sql(sql_index, conn, index_col = ['date'], parse_dates = ['date'])
    # conn.close()
    index_df.to_csv(symbol + "_data.csv")
    return index_df
if __name__ == "__main__":
    #df = load_index('000300', '20150101', '20150107')
    #print df
    print "main"
