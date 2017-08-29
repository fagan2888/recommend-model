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

commandA = 'select ra_date,ra_nav,ra_inc from ra_pool_nav where ra_pool = 92101 and ra_category = 0'
commandL = 'select ra_date,ra_nav,ra_inc from ra_pool_nav where ra_pool = 92101 and ra_category = 11'
commandS = 'select ra_date,ra_nav,ra_inc from ra_pool_nav where ra_pool = 92101 and ra_category = 12'
dataA = pd.read_sql(commandA,conn_aa)
dataL = pd.read_sql(commandL,conn_aa)
dataS = pd.read_sql(commandS,conn_aa)
datalist = pd.Series([dataA,dataL,dataS],index=[30290,30291,30292])
cptnames = pd.Series([u'线上股票池-全池',u'线上股票池-大盘',u'线上股票池-小盘'],index=[30290,30291,30292])

for assetid in datalist.index:
    try:
        command = u"insert into ra_composite_asset (globalid,ra_name,ra_calc_type,\
                    ra_begin_date,created_at,updated_at) values (" + str(assetid) + u",\
                    '"+cptnames[assetid]+"',3,'2013-01-01','2017-08-01','2017-08-01')"
        cur_aa.execute(command)
    except:
        pass
    data = datalist[assetid]
    for row in data.index:
        command = 'insert into ra_composite_asset_nav (ra_asset_id,ra_date,ra_nav,ra_inc) values (' \
                  + str(assetid) + ',"' + datetime.datetime.strftime(data.ix[row,'ra_date'],'%Y-%m-%d') + '",' \
                  + str(data.ix[row,'ra_nav']) + ',' + str(data.ix[row,'ra_inc']) + ')'
        cur_aa.execute(command)

cur_aa.close()
conn_aa.commit()
conn_aa.close()
