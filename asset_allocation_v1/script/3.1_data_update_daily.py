#coding=utf-8
import pandas as pd
import datetime
from dateindex import *
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
'''
ch_base = {"host": "rdsijnrreijnrre.mysql.rds.aliyuncs.com",
             "port": 3306,
             "user": "huyang",
             "passwd": "uMXfKx03Blg3jdPmauMx",
             "db":"caihui",
             "charset": "utf8"}

conn_ch  = MySQLdb.connect(**ch_base)
cur_ch   = conn_ch.cursor(MySQLdb.cursors.DictCursor)
conn_ch.autocommit(True)
'''
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

path1 = '1_dataCollection/'

try:
    method = str(sys.argv[1])
except:
    method = 'all'
try:
    startdate = datetime.datetime.strptime(str(sys.argv[2]),'%Y%m%d')
except:
    startdate = datetime.datetime.today()
try:
    enddate = datetime.datetime.strptime(str(sys.argv[3]),'%Y%m%d')
except:
    enddate = datetime.datetime.today()

tradeday = download('trade_dates','td_date').iloc[:,0]
tradeday = tradeday.apply(lambda x: '%04d' %x.year + '%02d' %x.month + '%02d' %x.day)

#fund data update
if method == 'fund' or method == 'all':
    for delta in range(0,(enddate-startdate).days+1):
        day = startdate + datetime.timedelta(days=delta)
        data = download('ra_fund_nav','ra_code ra_nav_adjusted ra_inc_adjusted',
                        pd.DataFrame([['ra_nav_date']]),pd.DataFrame([["'"+datetime.datetime.strftime(day,'%Y-%m-%d')+"'"]]))
        for row in data.index:
            try:
                olddata = pd.read_csv(path1+'fundData/price1d/'+str(int(data.ix[row,'ra_code']))+'.csv')
                olddata = dateindex(olddata)
                if day not in olddata.index:
                    newdata = pd.concat([olddata,pd.DataFrame([[data.ix[row,'ra_nav_adjusted']]],
                                                              index=[day],columns=[olddata.columns[0]])],
                                        axis=0).sort_index()
                    newdata.to_csv(path1+'fundData/price1d/'+str(int(data.ix[row,'ra_code']))+'.csv')
            except:
                newdata = pd.DataFrame([[data.ix[row,'ra_nav_adjusted']]],index=[day])
                newdata.to_csv(path1+'fundData/price1d/'+str(int(data.ix[row,'ra_code']))+'.csv')
            try:
                olddata = pd.read_csv(path1+'fundData/yield1d/'+str(int(data.ix[row,'ra_code']))+'.csv')
                olddata = dateindex(olddata)
                if day not in olddata.index:
                    newdata = pd.concat([olddata,pd.DataFrame([[data.ix[row,'ra_inc_adjusted']]],
                                                              index=[day],columns=[olddata.columns[0]])],
                                        axis=0).sort_index()
                    newdata.to_csv(path1+'fundData/yield1d/'+str(int(data.ix[row,'ra_code']))+'.csv')
            except:
                newdata = pd.DataFrame([[data.ix[row,'ra_inc_adjusted']]],index=[day])
                newdata.to_csv(path1+'fundData/yield1d/'+str(int(data.ix[row,'ra_code']))+'.csv')
        print day

#stock data update
if method == 'stock' or method == 'all':
    minid = pd.read_sql('select ID from TQ_SK_YIELDINDIC where TRADEDATE = "'\
                        +datetime.datetime.strftime(startdate,'%Y%m%d')+'" limit 1',conn_ch).iloc[0,0]
    data = pd.read_sql('select TRADEDATE,SYMBOL,SECODE,YIELD,YIELDM,YIELD3M,YIELD6M,YIELDY,TURNRATE,TURNRATEW,TURNRATEM,TURNRATE3M,TURNRATE6M,TURNRATEY \
                        from TQ_SK_YIELDINDIC where ID >= '+str(minid),conn_ch)
    data.ix[:,'SYMBOL'] = data.ix[:,'SYMBOL'].apply(lambda x: int(x))
    for day in data.ix[:,'TRADEDATE'].drop_duplicates():
        if day in list(tradeday):
            data.ix[data.ix[:,'TRADEDATE']==day,:].to_csv(path1+'stockData/TQ_SK_YIELDINDIC/'+day+'.csv')
            print day
'''
cur_ch.close()
conn_ch.commit()
conn_ch.close()
'''
cur_aa.close()
conn_aa.commit()
conn_aa.close()

cur_mf.close()
conn_mf.commit()
conn_mf.close()
