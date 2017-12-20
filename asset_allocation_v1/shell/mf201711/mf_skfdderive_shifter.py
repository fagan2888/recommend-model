#coding=utf-8
import pandas as pd
import datetime,time
import MySQLdb
import sys
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

def strs(x):
    try:
        return datetime.datetime.strftime(datetime.datetime.strptime(str(x),'%Y%m%d'),'%Y-%m-%d') #须谨防恰好可以表示为日期的数字出现
    except:
        try:
            return str(x)
        except:
            return x

def code6(x):
    try:
        return '0'*(6-len(str(int(x))))+str(int(x))
    except:
        return x

def writein2(code,date):
    values = pd.Series(map(lambda x: x.ix[code,date], datas),index=filelist).dropna().apply(lambda x: str(x))
    if len(list(values)) > 0:
        try:
            command = u"insert into " + chart + u" (fd_code,date," + ','.join(list(values.index)) + ",created_at,updated_at) values " \
                      + u"('%s','%s'," %(code,datetime.datetime.strftime(datetime.datetime.strptime(str(date),'%Y%m%d'),'%Y-%m-%d')) \
                      + ",".join(list(values)) \
                      + ",'%s','%s')" %(today,today) #无论如何default都会转为0，只能在建表时解决
            cur_aa.execute(command)
        except Exception as e:
            print e

aa_base = {"host": "127.0.0.1",
            "port": 3306,
            "user": "huyang",
            "passwd": "uMXfKx03Blg3jdPmauMx",
            "db":"asset_allocation",
            "charset": "utf8"}

conn_aa  = MySQLdb.connect(**aa_base)
cur_aa   = conn_aa.cursor(MySQLdb.cursors.DictCursor)
conn_aa.autocommit(True)

chart = 'mf_skfd_derivereport'

today = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d')
filelist = ['sharp_y','sharp_2y','sharp_5y',
            'sortino_y','sortino_2y','sortino_5y',
            'jenson_y','jenson_2y','jenson_5y']

datas = []
for csv in filelist:
    data = pd.read_csv('mf_%s.csv'%csv,index_col='SECODE').dropna(how='all',axis=0).dropna(how='all',axis=1)
    data = data.ix[data.index[1:data.shape[0]],:] #规避烦人的首行index为空错误
    data.index = map(lambda x: code6(x), data.index)
    datas += [data]
print 'Data loaded.'

codes = []
dates = []
for i in range(0,len(filelist)):
    codes = list(set(codes)|set(datas[i].index))
    dates = list(set(dates)|set(datas[i].columns))
for i in range(0,len(filelist)):
    datas[i] = datas[i].ix[codes,dates] #谨防之后找不到对应索引

for code in codes:
    for date in dates:
        writein2(code,date)
    print code

cur_aa.close()
conn_aa.commit()
conn_aa.close()
