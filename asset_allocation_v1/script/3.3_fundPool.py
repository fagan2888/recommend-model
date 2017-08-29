#coding=utf-8
import pandas as pd
import numpy as np
import datetime
import os
import statsmodels.formula.api as smf
import scipy.stats as stats
from scipy.stats import norm
import sys
from globalvalue import *

#生成至今为止的时间节点序列 #每年times次
times = 12
d90 = []
for year in range(2011,datetime.datetime.now().year):
    for month in range(0,times):
        d90 += [datetime.datetime(year,month*(12/times)+1,1)]
for month in range(0,(datetime.datetime.now().month-1)/(12/times)+1):
    d90 += [datetime.datetime(datetime.datetime.now().year,month*(12/times)+1,1)]
d90 = map(lambda x: x-datetime.timedelta(days=1),d90)

dreport = []
for year in range(2011,datetime.datetime.now().year):
    for month in range(0,2):
        dreport += [datetime.datetime(year,month*6+1,1)]
for month in range(0,(datetime.datetime.now().month-1)/6+1):
    dreport += [datetime.datetime(datetime.datetime.now().year,month*6+1,1)]
dreport = map(lambda x: x-datetime.timedelta(days=1),dreport)

#生成按持仓数据调仓时间
dstock = []
dfund = []
for year in range(2011,datetime.datetime.now().year):
    dstock += [datetime.datetime(year,5,1)]
    dstock += [datetime.datetime(year,9,1)]
    dfund += [datetime.datetime(year,4,1)]
    dfund += [datetime.datetime(year,9,1)]
if datetime.datetime.now().month >= 4:
    dfund += [datetime.datetime(datetime.datetime.now().year,4,1)]
if datetime.datetime.now().month >= 5:
    dstock += [datetime.datetime(datetime.datetime.now().year,5,1)]
if datetime.datetime.now().month >= 9:
    dstock += [datetime.datetime(datetime.datetime.now().year,9,1)]
    dfund += [datetime.datetime(datetime.datetime.now().year,9,1)]
dstock = map(lambda x: x-datetime.timedelta(days=1),dstock)
dfund = map(lambda x: x-datetime.timedelta(days=1),dfund)

def daystr(datetime):
    return str(datetime)[0:4]+str(datetime)[5:7]+str(datetime)[8:10]

#环境变量
path1 = '1_dataCollection/fundData/'
path2 = '2_factorTest/'
path3 = '3_fundPool/'
factorpath = path3 + 'fundFactorValue/'
validpath = path3 + 'validFund.csv'
outpath = path3 + 'pool/'
fundpath = path1 + 'yield1d/'
layerpath = path2 + 'layerYield1d.csv'

factorslist = factorslist_33
kindslist = kindslist_33
holdlist = holdlist_33

delayfactor = ['holder_avgpct L','holder_avgpct S','BP L','BP S']

#合法基金列表读取
valids = pd.read_csv(validpath)

#基金数据读取
fundList = pd.Series(os.listdir(fundpath)).apply(lambda x: x.replace('.csv',''))
rates = []
for fund in fundList:
    data = pd.read_csv(fundpath+fund+'.csv')
    rates += [data.set_index(data.columns[0])]
    print fund
rates = pd.concat(rates,axis=1)
rates.columns = fundList
rates.index = map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m-%d'), rates.index)
print 'Fund data loaded.'

#因子指数读取
indexes = pd.read_csv(layerpath)
indexes = indexes.set_index(indexes.columns[0])
try:
    indexes.index = map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m-%d'), indexes.index)
except:
    indexes.index = map(lambda x: datetime.datetime.strptime(str(x),'%Y/%m/%d'), indexes.index)
print 'Factor index data loaded.'

#基金筛选过程循环
for nameid in idrange:
    factors = factorslist[nameid]
    kinds = kindslist[nameid] #基金池中最多且尽量选取kinds只基金
    hold = holdlist[nameid] #风格持续性要求的宽限倍数
    if nameid == 0:
        nameid = ''
    else:
        nameid = str(nameid)

    #基金筛选过程
    inimonth = 8
    for factor in factors:
        value = pd.read_csv(factorpath+factor+'.csv').set_index('FSYMBOL')
        earnMatL = pd.DataFrame([[float('Nan')]*kinds]*len(d90),index=d90,columns=map(lambda x: 'fund'+str(x), range(1,kinds+1)))
        earnMatS = pd.DataFrame([[float('Nan')]*kinds]*len(d90),index=d90,columns=map(lambda x: 'fund'+str(x), range(1,kinds+1)))
        if factor in delayfactor:
            dfactor = dstock
        else:
            dfactor = dfund
        drep = pd.Series(dreport[0:len(dfactor)],index=dfactor)
        for date in range(inimonth,len(d90)):
            if d90[date] in dfactor:
                valueL = -(-value.ix[map(lambda x: int(str(x).replace('.0','')), valids.ix[:,daystr(drep[d90[date]])].dropna()),daystr(drep[d90[date]])]).dropna().sort_values()[0:kinds]
                if date > inimonth:
                    fundsL = list(set(earnMatL.ix[d90[date-1],:]).intersection(set(-(-value.ix[map(lambda x: int(str(x).replace('.0','')), valids.ix[:,daystr(drep[d90[date]])].dropna()),daystr(drep[d90[date]])]).dropna().sort_values()[0:int(hold*kinds)].index)))
                else:
                    fundsL = []
                fundsL = list(pd.Series(fundsL+list(valueL.index)).drop_duplicates()[0:kinds])
                valueS = value.ix[map(lambda x: int(str(x).replace('.0','')), valids.ix[:,daystr(drep[d90[date]])].dropna()),daystr(drep[d90[date]])].dropna().sort_values()[0:kinds]
                if date > inimonth:
                    fundsS = list(set(earnMatS.ix[d90[date-1],:]).intersection(set(value.ix[map(lambda x: int(str(x).replace('.0','')), valids.ix[:,daystr(drep[d90[date]])].dropna()),daystr(drep[d90[date]])].dropna().sort_values()[0:int(hold*kinds)].index)))
                else:
                    fundsS = []
                fundsS = list(pd.Series(fundsS+list(valueS.index)).drop_duplicates()[0:kinds])
                spareL = (-value.ix[map(lambda x: int(str(x).replace('.0','')), valids.ix[:,daystr(drep[d90[date]])].dropna()),daystr(drep[d90[date]])]).dropna().sort_values()[0:int(hold*kinds)].index
                spareS = value.ix[map(lambda x: int(str(x).replace('.0','')), valids.ix[:,daystr(drep[d90[date]])].dropna()),daystr(drep[d90[date]])].dropna().sort_values()[0:int(hold*kinds)].index
                earnMatL.ix[d90[date],:] = list(fundsL)
                earnMatS.ix[d90[date],:] = list(fundsS)
            else:
                rateL = rates.ix[(rates.index>d90[date-1])&(rates.index<=d90[date]),map(lambda x:str(x),spareL)]
                rateL = rateL.ix[rateL.count(axis=1)>0,:].dropna(axis=1)
                rateS = rates.ix[(rates.index>d90[date-1])&(rates.index<=d90[date]),map(lambda x:str(x),spareS)]
                rateS = rateS.ix[rateS.count(axis=1)>0,:].dropna(axis=1)
                indexL = indexes.ix[(indexes.index>d90[date-1])&(indexes.index<=d90[date]),factor+' L']-1
                indexS = indexes.ix[(indexes.index>d90[date-1])&(indexes.index<=d90[date]),factor+' S']-1
                corrL = np.corrcoef(pd.concat([indexL,rateL],axis=1).dropna(),rowvar=0)[0]
                corrL = pd.Series(corrL[1:len(corrL)],index=rateL.columns).dropna()
                corrS = np.corrcoef(pd.concat([indexS,rateS],axis=1).dropna(),rowvar=0)[0]
                corrS = pd.Series(corrS[1:len(corrS)],index=rateS.columns).dropna()
                #corrll = abs(np.corrcoef(pd.concat([indexL,rateL],axis=1).dropna(),rowvar=0)[0])
                #corrls = abs(np.corrcoef(pd.concat([indexS,rateL],axis=1).dropna(),rowvar=0)[0])
                #corrsl = abs(np.corrcoef(pd.concat([indexL,rateS],axis=1).dropna(),rowvar=0)[0])
                #corrss = abs(np.corrcoef(pd.concat([indexS,rateS],axis=1).dropna(),rowvar=0)[0])
                #corrL = (float(1)/pd.Series(corrll[1:len(corrll)],index=rateL.columns)*pd.Series(corrls[1:len(corrls)],index=rateL.columns)).dropna()
                #corrS = (float(1)/pd.Series(corrss[1:len(corrss)],index=rateS.columns)*pd.Series(corrsl[1:len(corrsl)],index=rateS.columns)).dropna()
                #spareL1 = corrL[corrL<=np.percentile(corrL,80)].index
                #spareS1 = corrS[corrS<=np.percentile(corrS,80)].index
                spareL1 = corrL[corrL>=np.percentile(corrL,20)].index
                spareS1 = corrS[corrS>=np.percentile(corrS,20)].index
                fundsL = list(set(earnMatL.ix[d90[date-1],:]).intersection(set(map(lambda x:int(x),spareL1))))
                fundsL = list(pd.Series(fundsL+list(map(lambda x:int(x),spareL1))).drop_duplicates()[0:kinds])
                fundsS = list(set(earnMatS.ix[d90[date-1],:]).intersection(set(map(lambda x:int(x),spareS1))))
                fundsS = list(pd.Series(fundsS+list(map(lambda x:int(x),spareS1))).drop_duplicates()[0:kinds])
                earnMatL.ix[d90[date],:] = list(fundsL)
                earnMatS.ix[d90[date],:] = list(fundsS)
            print date
        earnMatL.to_csv(outpath+'pool'+'-'+factor+' L'+nameid+'.csv',header=True)
        earnMatS.to_csv(outpath+'pool'+'-'+factor+' S'+nameid+'.csv',header=True)
        print factor
