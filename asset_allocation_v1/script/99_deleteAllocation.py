#coding=utf-8
import pandas as pd
import datetime
from dateindex import *
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

for nameid in idrange:
    assetid = str(30200 + nameid)
    command = 'delete from ra_composite_asset_position where ra_asset_id = ' + assetid
    cur_aa.execute(command)
    print assetid + ' deleted.'

cur_aa.close()
conn_aa.commit()
conn_aa.close()
