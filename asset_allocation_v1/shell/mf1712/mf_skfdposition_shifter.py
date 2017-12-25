#coding=utf-8
import pandas as pd
import datetime,time
import MySQLdb
import sys
import multiprocessing

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
        return x+201
    elif str(x)[4:8] == '1231':
        return x+9100
    else:
        return None

def fdcodemap(x):
    try:
        return pd.Series(fdcode.ix[x])[0]
    except:
        return None

def skcodemap(x):
    try:
        return pd.Series(skcode.ix[x])[0]
    except:
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

chart = 'mf_skfd_position'

today = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d')
data = pd.read_csv('mf_skfd_position.csv',index_col=0).dropna()
data.columns = ['report_date','fd_code','sk_code','position']
data.ix[:,'publish_date'] = data.ix[:,'report_date'].apply(reportdate)
fdcode = pd.read_csv('mf_fundcode.csv',index_col=0).set_index('SECODE').ix[:,'FSYMBOL']
fdcode = fdcode.apply(lambda x: 'FD.%s' %code6(x))
skcode = pd.read_csv('mf_stockcode.csv',index_col=0).set_index('SECODE').ix[:,'SYMBOL']
skcode = skcode.apply(lambda x: 'SK.%s' %code6(x))
data.ix[:,'fd_code'] = data.ix[:,'fd_code'].apply(fdcodemap)
data.ix[:,'sk_code'] = data.ix[:,'sk_code'].apply(skcodemap)
data = data.dropna()

for row in data.index:
    try:
        command = u"insert into " + chart + u" (" + ','.join(list(data.columns)) \
                  + u",created_at,updated_at) values ('" \
                  + "','".join(map(lambda x: strs(x), data.ix[row,:])) \
                  + u"','" + today + u"','" + today + u"')"
        cur_aa.execute(command)
        print row
    except Exception as e:
        print e

cur_aa.close()
conn_aa.commit()
conn_aa.close()
