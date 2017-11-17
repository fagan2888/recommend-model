#coding=utf-8
import pandas as pd
import datetime
import MySQLdb

aa_base = {"host": "127.0.0.1",
             "port": 3306,
             "user": "huyang",
             "passwd": "uMXfKx03Blg3jdPmauMx",
             "db":"asset_allocation",
             "charset": "utf8"}

conn_aa  = MySQLdb.connect(**aa_base)
cur_aa   = conn_aa.cursor(MySQLdb.cursors.DictCursor)
conn_aa.autocommit(True)

assetid = 30300
data = pd.read_csv('backMat.csv',index_col=0,parse_dates=[0])

for date in data.index:
    command = 'insert into ra_composite_asset_nav (ra_asset_id,ra_date,ra_nav,ra_inc) values (' \
              + str(assetid) + ',"' + datetime.datetime.strftime(date,'%Y-%m-%d') + '",' \
              + str(data.ix[date,'price']) + ',' + str(data.ix[date,'rate']) + ')'
    cur_aa.execute(command)
    print date

cur_aa.close()
conn_aa.commit()
conn_aa.close()
