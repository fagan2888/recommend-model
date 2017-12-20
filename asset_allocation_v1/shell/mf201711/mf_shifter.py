#coding=utf-8
import pandas as pd
import datetime
import MySQLdb
import sys

def strs(x):
    try:
        return datetime.datetime.strftime(datetime.datetime.strptime(str(x),'%Y%m%d'),'%Y-%m-%d') #须谨防恰好可以表示为日期的数字出现
    except:
        try:
            return str(x)
        except:
            return x

aa_base = {"host": "127.0.0.1",
            "port": 3306,
            "user": "huyang",
            "passwd": "uMXfKx03Blg3jdPmauMx",
            "db":"asset_allocation",
            "charset": "utf8"}

conn_aa  = MySQLdb.connect(**aa_base)
cur_aa   = conn_aa.cursor(MySQLdb.cursors.DictCursor)
conn_aa.autocommit(True)

today = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d')
chart = sys.argv[1]

data = pd.read_csv(chart+'.csv',encoding='gb2312').fillna('')
for row in data.index:
    try:
        command = u"insert into " + chart + u" (" + ','.join(list(data.columns)) \
                  + u",created_at,updated_at) values ('" \
                  + "','".join(map(lambda x: strs(x), data.ix[row,:])) \
                  + u"','" + today + u"','" + today + u"')"
        cur_aa.execute(command)
    except Exception as e:
        print e

cur_aa.close()
conn_aa.commit()
conn_aa.close()
