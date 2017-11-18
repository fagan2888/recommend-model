#coding=utf-8
import pandas as pd
import datetime
import MySQLdb
import os
import sys

mf_base = {"host": "127.0.0.1",
             "port": 3306,
             "user": "huyang",
             "passwd": "uMXfKx03Blg3jdPmauMx",
             "db":"mofang",
             "charset": "utf8"}

conn_mf  = MySQLdb.connect(**mf_base)
cur_mf   = conn_mf.cursor(MySQLdb.cursors.DictCursor)
conn_mf.autocommit(True)

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
nameid = u'多因子净值'
try:
    command = u"insert into ra_composite_asset (globalid,ra_name,ra_calc_type,\
                ra_begin_date,created_at,updated_at) values (" + str(assetid) + u",\
                '"+nameid+"',3,'2011-09-01','2017-11-31','2017-11-31')"
    cur_aa.execute(command)
except Exception as e:
    print e

def download(listname,cols,fliters=pd.DataFrame(),values=pd.DataFrame()):
    if fliters.shape[0]:
        command = "select "+cols.replace(' ',',')+" from "+listname+" where ( "+str(fliters.iloc[0,0])+" = "+str(values.iloc[0,0])
        for j in range(1,fliters.shape[1]):
            command = command + " and "+str(fliters.iloc[0,j])+" = "+str(values.iloc[0,j])
        command = command + " )"
        for i in range(1,fliters.shape[0]):
            command = command + " or ("+str(fliters.iloc[i,0])+" = "+str(values.iloc[i,0])
            for j in range(1,fliters.shape[1]):
                command = command + " and "+str(fliters.iloc[i,j])+" = "+str(values.iloc[i,j])
            command = command + " )"
    else:
        command = "select "+cols.replace(' ',',')+" from "+listname
    return pd.read_sql(command,conn_mf)

data = pd.read_csv('fundpool.csv',index_col=0,parse_dates=[0])
for date in data.index:
    asset = data.ix[date,:].dropna()
    for fund in asset.index:
        code = '0'*(6-len(str(int(fund.replace('FD.',''))))) + str(int(fund.replace('FD.','')))
        codeid = download('fund_infos','fi_globalid',pd.DataFrame([['fi_code']]),pd.DataFrame([[code]])).iloc[0,0]
        command = "insert into ra_composite_asset_position \
                    (ra_asset_id,ra_date,ra_fund_id,ra_fund_code,ra_fund_type,ra_fund_ratio) values \
                    (" + str(assetid) + ",'" + datetime.datetime.strftime(date,'%Y-%m-%d') + \
                    "'," + str(codeid) + ",'" + code + "',1," + str(data.ix[date,fund]) + ")"
        cur_aa.execute(command)
    print date,'adjusted.'

cur_aa.close()
conn_aa.commit()
conn_aa.close()

cur_mf.close()
conn_mf.commit()
conn_mf.close()
