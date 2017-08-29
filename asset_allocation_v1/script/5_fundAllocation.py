#coding=utf-8
import pandas as pd
import datetime
from dateindex import *
import MySQLdb
import os
import sys
from globalvalue import *

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

path3 = '3_fundPool/'
path4 = '4_factorTiming/'
path5 = '5_allocationResult/'

try:
    startdate = datetime.datetime.strptime(str(sys.argv[1]),'%Y%m%d')
except:
    startdate = datetime.datetime.today()
try:
    enddate = datetime.datetime.strptime(str(sys.argv[2]),'%Y%m%d')
except:
    enddate = datetime.datetime.today()

for nameid in idrange:
    assetid = 30200 + int(nameid)
    if nameid == 0:
        nameid = ''
    else:
        nameid = str(nameid)

    ratio = pd.read_csv(path4+'position'+nameid+'.csv')
    ratio = dateindex(ratio)

    for delta in range(0,(enddate-startdate).days+1):
        change = False
        day = startdate + datetime.timedelta(days=delta)
        dayratio = ratio.ix[max(ratio.index[ratio.index<day]),:]
        
        validfactor = pd.Series([0]*len(dayratio),index=dayratio.index)
        for i in range(0,len(dayratio)):
            if dayratio[i] > 0:
                try:
                    pd.read_csv(path3+'pool/pool-'+dayratio.index[i]+nameid+'.csv')
                    validfactor[i] = 1
                except:
                    pass
        dayratio = dayratio*validfactor / sum(dayratio*validfactor)
        
        if (day - max(ratio.index[ratio.index<day])).days == 1:
            change = True

        position = []
        
        for i in range(0,len(dayratio)):
            if dayratio[i] > 0:
                pool = pd.read_csv(path3+'pool/pool-'+dayratio.index[i]+nameid+'.csv')
                pool = dateindex(pool)
                if (day - max(pool.index[pool.index<day])).days == 1:
                    change = True
                pool = pool.ix[max(pool.index[pool.index<day]),:]
                pos = pd.Series([float(dayratio[i])/len(pool)]*len(pool),index=pool.values)
                position += [pos]

        position = pd.concat(position,axis=0)
        position = position.groupby(position.index).sum().rename('allocation')

        try:
            os.mkdir(path5+str(assetid))
        except:
            pass
        position.to_csv(path5+str(assetid)+'/'+datetime.datetime.strftime(day,'%Y%m%d')+'.csv',header=True)
        
        if change:
            for fund in position.index:
                fund6 = '0'*(6-len(str(int(fund)))) + str(int(fund))
                fundid = download('fund_infos','fi_globalid',pd.DataFrame([['fi_code']]),pd.DataFrame([[fund6]])).iloc[0,0]
                command = "insert into ra_composite_asset_position \
                            (ra_asset_id,ra_date,ra_fund_id,ra_fund_code,ra_fund_type,ra_fund_ratio) values \
                            (" + str(assetid) + ",'" + datetime.datetime.strftime(day-datetime.timedelta(days=1),'%Y-%m-%d') + \
                            "'," + str(fundid) + ",'" + fund6 + "',1," + str(position[fund]) + ")"
                cur_aa.execute(command)
            print 'adjusted.'
        
        print day

cur_aa.close()
conn_aa.commit()
conn_aa.close()

cur_mf.close()
conn_mf.commit()
conn_mf.close()
