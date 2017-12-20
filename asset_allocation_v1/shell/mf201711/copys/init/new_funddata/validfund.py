#coding=utf8

import pandas as pd
import numpy as np
import datetime
import os,sys

locpath = sys.path[0]+'/'

try:
    holdlimit = float(sys.argv[1])
    sizelimit = float(sys.argv[2])
    alphalimit = float(sys.argv[3])
    alphakind = int(sys.argv[4])
except:
    holdlimit = 2
    sizelimit = 2
    alphalimit = 0
    alphakind = 1

#生成至今为止的时间节点序列 1997.12.31 - 2017.3.31
d90 = []
for year in range(2011,datetime.datetime.now().year):
    for month in range(0,12):
        d90 += [datetime.datetime(year,month+1,1)]
for month in range(0,datetime.datetime.now().month):
    d90 += [datetime.datetime(datetime.datetime.now().year,month+1,1)]
d90 = map(lambda x: datetime.datetime.strftime(x-datetime.timedelta(days=1),'%Y%m%d'),d90)
d90 = d90[8:len(d90)]

def change_index(x):
    if '0331' in str(x):
        return str(int(x) + 84) #0415
    if '0630' in str(x):
        return str(int(x) + 201) #0831
    if '0930' in str(x):
        return str(int(x) + 86) #1015
    if '1231' in str(x):
        return str(int(x) + 9100) #10331

dirpath = '/home/huyang/MultiFactors201710/new_funddata/'

tipo = pd.read_csv(dirpath+u'tipoValidFund.csv') #上市时间限定（原合法基金筛选结果）
tipo.columns = map(lambda x: change_index(x), tipo.columns) #对数据获取时间进行校正

hold = pd.read_csv(dirpath+'fund_inshold.csv',index_col='SECODE')
size = pd.read_csv(dirpath+'fund_size.csv',index_col='SECURITYID')
hold.columns = map(lambda x: change_index(x), hold.columns) #对数据获取时间进行校正
size.columns = map(lambda x: change_index(x), size.columns) #对数据获取时间进行校正
print 'hold/size data read!'

alphas = ['fund_Sortino52W.csv','fund_Sortino24W.csv','fund_Sortino60W.csv',
          'fund_JENSON52W.csv','fund_JENSON24W.csv','fund_JENSON60W.csv',
          'fund_SHARP52W.csv','fund_SHARP24M.csv','fund_SHARP60M.csv']
alpha = pd.read_csv(dirpath+alphas[alphakind-1],index_col='SECODE') #改文件名决定alpha用什么指标
alpha.index = map(lambda x: str(x).replace('.0',''), alpha.index)
alpha = alpha.ix[alpha.index!='nan',:]
print 'alpha data read!'

dates = list(pd.Series(list(set(tipo.columns)|set(hold.columns)|set(size.columns)|set(alpha.columns)|set(d90))).sort_values())
tipo = tipo.ix[:,dates].fillna(axis=1,method='pad').ix[:,d90]
hold = hold.ix[:,dates].fillna(axis=1,method='pad').ix[:,d90]
size = size.ix[:,dates].fillna(axis=1,method='pad').ix[:,d90]
alpha = alpha.ix[:,dates].fillna(axis=1,method='pad').ix[:,d90]

def intx(x):
    try:
        return int(x)
    except:
        return None

vfMat = pd.DataFrame()
for month in d90:
    tipocode = tipo.ix[:,month].dropna().apply(lambda x: str(x).replace('.0','')).values #通过这个还能筛掉非股基金
    holdcode = map(lambda x: str(intx(x)), hold.ix[(hold.ix[:,month]-hold.ix[tipocode,month].dropna().mean())/hold.ix[tipocode,month].dropna().std()<-holdlimit,month].index) #不要机构持仓过小的
    sizecode = map(lambda x: str(intx(x)), size.ix[(size.ix[:,month].apply(lambda x:np.log(x))-size.ix[tipocode,month].dropna().apply(lambda x:np.log(x)).mean())/size.ix[tipocode,month].dropna().apply(lambda x:np.log(x)).std()>sizelimit,month].index) #不要ln规模过大的
    alphacode = map(lambda x: str(intx(x)), alpha.ix[(alpha.ix[:,month]-alpha.ix[tipocode,month].dropna().mean())/alpha.ix[tipocode,month].dropna().std()<-alphalimit,month].index) #不要alpha指标过小的
    validfund = pd.Series(list(set(tipocode)-set(holdcode)-set(sizecode)-set(alphacode)))
    validfund = validfund.apply(lambda x: intx(x)).dropna()
    vfMat = vfMat.append(validfund.rename(month))
    print month, len(validfund)
vfMat.T.to_csv(locpath+'validfund.csv')
