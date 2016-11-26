#coding=utf8


import pandas as pd
import numpy  as np
from datetime import datetime
import MySQLdb


'''
db_params = {
            "host": "rdsijnrreijnrre.mysql.rds.aliyuncs.com",
            "port": 3306,
            "user": "koudai",
            "passwd": "Mofang123",
            "db":"caihui",
            "charset": "utf8"
        }
'''

db_base = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"mofang",
    "charset": "utf8"
}


if __name__ == '__main__':


    conn  = MySQLdb.connect(**db_base)
    cur   = conn.cursor()

    #df = pd.read_csv('yinhe_type.csv')
    lines = open('yinhe_type.csv').readlines()
    codes = []
    for line in lines:
        code = line.strip()
        code = '%06d' % (int)(code)
        codes.append(code)

    for code in codes:
        sql = 'select ra_corr from ra_corr_fund where ra_corr_id = 520002 and ra_fund_code= %s' % code
        try:
            cur.execute(sql)
            record = cur.fetchone()
            print code, ',' ,record[0]
        except:
            print code
