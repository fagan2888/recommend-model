#coding=utf-8
import pandas as pd
import numpy as np
import datetime
import MySQLdb
import sys
import os
import re

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

datapath = u'/home/huyang/MultiFactors201710/cleanedData/' #数据表位置

#'''
#读取因子列表
command = "select factor_name from mf_sk_factors where factor_source = 1" #把财报数据用读表格方式上传
factors = pd.read_sql(command,conn_aa).dropna().ix[:,'factor_name'].sort_values()
#'''

#多线程并行
splitnum = int(sys.argv[1])
splitloc = int(sys.argv[2])
splitstep = len(factors)/splitnum
if splitloc == splitnum-1:
    splitrange = '%s ~ %s' %(str(splitstep*splitloc),len(factors)-1)
    factors = factors.iloc[splitstep*splitloc:len(factors)]
else:
    splitrange = '%s ~ %s' %(str(splitstep*splitloc),str(splitstep*(splitloc+1)-1))
    factors = factors.iloc[splitstep*splitloc:splitstep*(splitloc+1)]
print 'group%s: %s' %(splitloc,','.join(list(factors)))

#'''
#逐因子上传因子值
for factor in factors:
    dates = map(lambda x: x.replace('.csv',''), os.listdir(datapath+factor+'/'))
    #逐日期按数据表上传因子值
    for date in dates:
        data = pd.read_csv(datapath+factor+'/'+date+'.csv').drop_duplicates()
        data.columns = ['code','value']
        ''' #原逐条插入代码
        for row in data.index:
            if data.ix[row,'value'] != np.inf and data.ix[row,'value'] != -np.inf and data.ix[row,'value'] != np.nan:
                command = "insert into mf_sk_factorvalue (periods_date,sk_code,factor_name,factor_value,created_at,updated_at) values ('%s','%s','%s',%s,'%s','%s')" \
                          %(datetime.datetime.strftime(datetime.datetime.strptime(date,'%Y%m%d'),'%Y-%m-%d'), #periods_date
                            'SK.'+'0'*(6-len(str(int(data.ix[row,'code']))))+str(int(data.ix[row,'code'])), #sk_code
                            factor, #factor_name
                            str(round(data.ix[row,'value'],6)), #factor_value
                            today, today) #created_at,updated_at
                cur_aa.execute(command)
        '''
        data = data[(data!=np.inf)&(data!=-np.inf)&(data!=np.nan)]
        if len(data):
            command = "insert into mf_sk_factorvalue (periods_date,sk_code,factor_name,factor_value,created_at,updated_at) values (%s);" \
                      %("),(".join(map(lambda row: "'%s','%s','%s',%s,'%s','%s'" \
                                       %(datetime.datetime.strftime(datetime.datetime.strptime(date,'%Y%m%d'),'%Y-%m-%d'), #periods_date
                                         'SK.'+'0'*(6-len(str(int(data.ix[row,'code']))))+str(int(data.ix[row,'code'])), #sk_code
                                         factor, #factor_name
                                         str(round(data.ix[row,'value'],6)), #factor_value
                                         today, today), #created_at,updated_at
                                       data.index))) #整理为一个命令批量插入
            cur_aa.execute(command)
        print '%s data shifted: factor %s' %(date,factor)
    #print '%s data shifted.' %factor
#'''

#断开数据库
cur_aa.close()
conn_aa.commit()
conn_aa.close()
