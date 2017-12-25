#coding=utf-8
import pandas as pd
import datetime,time
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

def reportdate(x):
    if str(x)[4:8] == '0630':
        return str(int(x)+201)
    elif str(x)[4:8] == '1231':
        return str(int(x)+9100)
    elif str(x)[4:8] == '0331':
        return str(int(x)+84)
    elif str(x)[4:8] == '0930':
        return str(int(x)+85)
    else:
        return None

def code6(x):
    try:
        return '0'*(6-len(str(int(x))))+str(int(x))
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

chart = 'mf_skfd_marketreport'

today = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d')
size = pd.read_csv('mf_fundsize.csv',index_col='SECURITYID')
size.index = map(lambda x: code6(x), size.index)
hold = pd.read_csv('mf_fundhold.csv',index_col='SECODE')
hold.index = map(lambda x: code6(x), hold.index)

codes = list(set(size.index).intersection(set(hold.index)))
reportdates = list(set(size.columns).intersection(set(hold.columns)))
publishdates = map(lambda x: reportdate(x), reportdates)
for di in range(0,len(reportdates)):
    date = reportdates[di]
    pdate = publishdates[di]
    for code in codes:
        if not pd.isnull(size.ix[code,date]) and not pd.isnull(hold.ix[code,date]):
            try:
                command = u"insert into " + chart + u" (fd_code,report_date,publish_date,size,hold,created_at,updated_at) values " \
                          u"('%s','%s','%s',%s,%s,'%s','%s')" \
                          %(code,
                            datetime.datetime.strftime(datetime.datetime.strptime(str(date),'%Y%m%d'),'%Y-%m-%d'),
                            datetime.datetime.strftime(datetime.datetime.strptime(str(pdate),'%Y%m%d'),'%Y-%m-%d'),
                            size.ix[code,date],
                            hold.ix[code,date],
                            today,today)
                cur_aa.execute(command)
                print date,code
            except Exception as e:
                print e

cur_aa.close()
conn_aa.commit()
conn_aa.close()
