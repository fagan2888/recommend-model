#coding=utf-8
import MySQLdb
import os
import sys
from globalvalue import *

aa_base = {"host": "127.0.0.1",
             "port": 3306,
             "user": "huyang",
             "passwd": "uMXfKx03Blg3jdPmauMx",
             "db":"asset_allocation",
             "charset": "utf8"}

conn_aa  = MySQLdb.connect(**aa_base)
cur_aa   = conn_aa.cursor(MySQLdb.cursors.DictCursor)
conn_aa.autocommit(True)

names = names_0

for nameid in idrange:
    assetid = 30200 + nameid
    try:
        command = u"insert into ra_composite_asset (globalid,ra_name,ra_calc_type,\
                    ra_begin_date,created_at,updated_at) values (" + str(assetid) + u",\
                    '"+names[nameid]+"',3,'2013-01-01','2017-08-01','2017-08-01')"
        cur_aa.execute(command)
    except Exception as e:
        print e

cur_aa.close()
conn_aa.commit()
conn_aa.close()
