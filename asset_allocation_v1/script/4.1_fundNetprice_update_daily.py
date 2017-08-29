#coding=utf-8
import pandas as pd
import numpy as np
import datetime
import sys
import MySQLdb
import pandas as pd
import os
import progressbar

db_base = {"host": "127.0.0.1",
             "port": 3306,
             "user": "huyang",
             "passwd": "uMXfKx03Blg3jdPmauMx",
             "db":"mofang",
             "charset": "utf8"}

conn  = MySQLdb.connect(**db_base)
cur   = conn.cursor(MySQLdb.cursors.DictCursor)
conn.autocommit(True)

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
    return pd.read_sql(command,conn)

path3 = '3_fundPool/'
path4 = '4_factorTiming/'

factor = ['ln_capital','BP','std_3m','tradevolumn_3m','holder_avgpct']
factors = map(lambda x: x+' L', factor) + map(lambda x: x+' S', factor)

try:
    date = datetime.datetime.strptime(sys.argv[1],'%Y%m%d')
except:
    date = datetime.datetime.today()

def dateindex(data):
    data = data.set_index(data.columns[0])
    try:
        data.index = map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'), data.index)
    except:
        try:
            data.index = map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'), data.index)
        except:
            data.index = map(lambda x: datetime.datetime.strptime(x,'%Y%m%d'), data.index)
    return data

def update_fundRate(date):
    rates = []
    progress = progressbar.ProgressBar()
    for i in progress(range(0,len(factors))):
        data = dateindex(pd.read_csv(path3+'pool/pool-'+factors[i]+'.csv'))
        adjustdate = max(data.index[data.index<date])
        funds = map(lambda x: '0'*(6-len(str(int(x))))+str(int(x)), data.ix[adjustdate,:].values)
        fliters = pd.DataFrame([['ra_date','ra_code']]*len(funds))
        values = pd.concat([pd.Series(["'"+datetime.datetime.strftime(date,'%Y-%m-%d')+"'"]*len(funds)),pd.Series(funds)],axis=1)
        rate = download("ra_fund_nav","ra_inc_adjusted",fliters,values).iloc[0,0]
        rates += [1+np.mean(rate)]
    rates = pd.Series(rates,index=factors)
    return rates

def update_fundNetprice(date=datetime.datetime.today()):
    netprice = dateindex(pd.read_csv(path4+'fundNetprice.csv'))
    lastday = netprice.index[netprice.shape[0]-1]
    tradeday = download('trade_dates','td_date').max().max()
    tradeday = datetime.datetime(tradeday.year,tradeday.month,tradeday.day)
    for delta in range(1,(min(date,tradeday)-lastday).days):
        day = lastday + datetime.timedelta(days=delta)
        rates = update_fundRate(day)[netprice.columns]
        netprice = pd.concat([netprice,pd.DataFrame([(netprice.iloc[netprice.shape[0]-1,:]*rates).values],columns=netprice.columns)],axis=0)
        netprice.index = list(netprice.index[0:(netprice.shape[0]-1)])+[day]
        netprice.to_csv(path4+'fundNetprice.csv',header=True)
        print day

update_fundNetprice(date)

cur.close()
conn.commit()
conn.close()
