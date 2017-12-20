#coding=utf-8
import pandas as pd
import numpy as np
import datetime
import MySQLdb
import sys
import os
import re

#解决令人发指的rolling后无法处理带nan的.mean(),.std(),.sum()的问题的函数
def rollingmean(seri,window):
    result = pd.Series(index=seri.index)
    for i in range(0,len(seri)):
        try:
            result[result.index[i]] = seri.iloc[(i-window+1):(i+1)].dropna().mean()
        except:
            result[result.index[i]] = None
    return result

def rollingstd(seri,window):
    result = pd.Series(index=seri.index)
    for i in range(0,len(seri)):
        try:
            result[result.index[i]] = seri.iloc[(i-window+1):(i+1)].dropna().std()
        except:
            result[result.index[i]] = None
    return result

def rollingsum(seri,window):
    result = pd.Series(index=seri.index)
    for i in range(0,len(seri)):
        try:
            result[result.index[i]] = seri.iloc[(i-window+1):(i+1)].dropna().sum()
        except:
            result[result.index[i]] = None
    return result

#定义计算区间/报告期 #从1990年起
periods = pd.date_range(start=datetime.datetime(1990,1,1),
                        end=datetime.datetime.now(),freq='M')
today = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d')

#连接数据库
aa_base = {"host": "127.0.0.1",
            "port": 3306,
            "user": "huyang",
            "passwd": "uMXfKx03Blg3jdPmauMx",
            "db":"asset_allocation",
            "charset": "utf8"}

conn_aa  = MySQLdb.connect(**aa_base)
cur_aa   = conn_aa.cursor(MySQLdb.cursors.DictCursor)
conn_aa.autocommit(True)

mf_base = {"host": "127.0.0.1",
             "port": 3306,
             "user": "huyang",
             "passwd": "uMXfKx03Blg3jdPmauMx",
             "db":"mofang",
             "charset": "utf8"}

conn_mf  = MySQLdb.connect(**mf_base)
cur_mf   = conn_mf.cursor(MySQLdb.cursors.DictCursor)
conn_mf.autocommit(True)

#'''
#读取股票列表
command = "select globalid from ra_stock_nav"
stocks = pd.read_sql(command,conn_mf).drop_duplicates().ix[:,'globalid'].sort_values()
#print 'stocks amount is: %s' %str(stocks.shape[0])
#'''

#多线程并行
splitnum = int(sys.argv[1])
splitloc = int(sys.argv[2])
splitstep = len(stocks)/splitnum
if splitloc == splitnum-1:
    splitrange = '%s ~ %s' %(str(splitstep*splitloc),len(stocks)-1)
    stocks = stocks.iloc[splitstep*splitloc:len(stocks)]
else:
    splitrange = '%s ~ %s' %(str(splitstep*splitloc),str(splitstep*(splitloc+1)-1))
    stocks = stocks.iloc[splitstep*splitloc:splitstep*(splitloc+1)]
print 'group%s: %s' %(splitloc,splitrange)

#'''
#读取因子列表
command = "select factor_name,factor_source,formula from mf_sk_factors"
factors = pd.read_sql(command,conn_aa).dropna().set_index('factor_name')
factors = factors[factors.ix[:,'formula']!=''] #谨防数据库存了''而非空值
fields = factors.ix[:,'formula'].apply(lambda x: re.findall(r'\[.+?\]|\{.+?\}',x)) #匹配公式中所有字段
fields = pd.Series(sum(list(fields),[])).apply(lambda x: re.sub('\[|\]|\{|\}','',x)).drop_duplicates() #字段去[]{}去重
#'''

#读取数据表字段名
command = "select COLUMN_NAME from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME = 'ra_stock_nav'"
navname = set(pd.read_sql(command,conn_mf).ix[:,'COLUMN_NAME'])
command = "select COLUMN_NAME from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME = 'ra_stock_yieldindic'"
ydcname = set(pd.read_sql(command,conn_mf).ix[:,'COLUMN_NAME']) - navname #要去重，这点很重要

#'''
#逐股票计算因子值
#stocks = stocks[stocks.index[0:2]]
for stock in stocks:

    #读取数据库股票数据
    command = "select sk_tradedate,%s from ra_stock_nav where globalid = '%s'" \
              %(','.join(list(navname.intersection(set(fields)))),stock)
    nav = pd.read_sql(command,conn_mf).set_index('sk_tradedate') #读取nav表
    command = "select sk_tradedate,%s from ra_stock_yieldindic where globalid = '%s'" \
              %(','.join(list(ydcname.intersection(set(fields)))),stock)
    ydc = pd.read_sql(command,conn_mf).set_index('sk_tradedate') #读取yieldindic表
    dates = pd.date_range(start=min(nav.index), end=max(nav.index))
    data = pd.concat([nav,ydc],axis=1).reindex(dates.values) #含nan的数据表，用于计算区间数据
    datapad = data.fillna(method='pad') #不含nan的数据表，用于计算即时数据
    #print '%s data loaded.' %stock
    #'''
    #逐因子按公式计算因子值
    for factor in factors.index:
        formula = factors.ix[factor,'formula'].replace("[","data.ix[:,'").replace("]","']") #处理区间数据
        formula = formula.replace("{","datapad.ix[:,'").replace("}","']") #处理即时数据
        formula = "(%s)[periods].dropna()" %formula
        fvalue = eval(formula) #执行公式
        ''' #原逐条插入代码
        for row in fvalue.index:
            if fvalue[row] != np.inf and fvalue[row] != -np.inf and fvalue[row] != np.nan:
                command = "insert into mf_sk_factorvalue (periods_date,sk_code,factor_name,factor_value,created_at,updated_at) values ('%s','%s','%s',%s,'%s','%s')" \
                          %(datetime.datetime.strftime(row,'%Y-%m-%d'), #periods_date
                            stock, #sk_code
                            factor, #factor_name
                            str(round(fvalue[row],6)), #factor_value
                            today, today) #created_at,updated_at
                cur_aa.execute(command)
        '''
        fvalue = fvalue[(fvalue!=np.inf)&(fvalue!=-np.inf)&(fvalue!=np.nan)]
        if len(fvalue):
            command = "insert into mf_sk_factorvalue (periods_date,sk_code,factor_name,factor_value,created_at,updated_at) values (%s);" \
                      %("),(".join(map(lambda row: "'%s','%s','%s',%s,'%s','%s'" \
                                       %(datetime.datetime.strftime(row,'%Y-%m-%d'), #periods_date
                                         stock, factor, #sk_code,factor_name
                                         str(round(fvalue[row],6)), #factor_value
                                         today, today), #created_at,updated_at
                                       fvalue.index))) #整理为一个命令批量插入
            cur_aa.execute(command)
        #print '%s data calculated: factor %s' %(stock,factor)
    print '%s data calculated.' %stock
    #'''
#'''

#断开数据库
cur_mf.close()
conn_mf.commit()
conn_mf.close()

cur_aa.close()
conn_aa.commit()
conn_aa.close()
