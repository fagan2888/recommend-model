#coding=utf-8
import pandas as pd
import numpy as np
import datetime
import os
import statsmodels.formula.api as smf
import scipy.stats as stats

#生成至今为止的时间节点序列
d90 = []
for year in range(2010,datetime.datetime.now().year):
    for month in range(0,4):
        d90 += [datetime.datetime(year,month*3+1,1)]
for month in range(0,(datetime.datetime.now().month-1)/3+1):
    d90 += [datetime.datetime(datetime.datetime.now().year,month*3+1,1)]
d90 += [datetime.datetime.today()]
d90 = map(lambda x: x-datetime.timedelta(days=1),d90)

def daystr(datetime):
    return str(datetime)[0:4]+str(datetime)[5:7]+str(datetime)[8:10]

#环境变量，因子名称赋值
path1 = '1_dataCollection/'
path2 = '2_factorTest/'
datepath = path1 + 'stockData/TQ_SK_YIELDINDIC/'
datapath = path1 + 'FcleanedData_standarded/'

factors = ['ln_capital','tradevolumn_3m','std_3m','holder_avgpct','BP']
afterwards = pd.Series([0,0,0,0,0],index=factors)

delayfactor = ['holder_avgpct','BP']
delaytime = [datetime.timedelta(30),datetime.timedelta(62),datetime.timedelta(31),datetime.timedelta(121)]
#财报数据：一季报4.30出，半年报8.31出，三季报10.31出，年报4.30出

#因子数据读取
datas = []
for factor in factors:
    dataone = [[]]*afterwards[factor]
    for i in range(afterwards[factor],len(d90)-2):
        try:
            data = pd.read_csv(datapath+factor+'/'+daystr(d90[i+1])+'.csv')
            datatmp = data[data.columns[1]].apply(lambda x: float(x))
            datatmp.index = data[data.columns[0]].apply(lambda x: str(x))
            dataone += [datatmp]
        except:
            dataone += [datatmp]
    datas += [dataone]
datas = pd.Series(datas,index=factors)
print 'Factor data loaded.'

#行情数据读取
tradeDates = pd.Series(os.listdir(datepath)).apply(lambda x: datetime.datetime.strptime(str(x).replace('.csv',''),'%Y%m%d')).sort_values()
tradeMat = []
for date in tradeDates:
    data = pd.read_csv(datepath+daystr(date)+'.csv')
    rate = data.iloc[:,4].fillna(0)/100+1
    rate.index = data.iloc[:,2].apply(lambda x: str(x))
    tradeMat += [rate]
    print date
tradeMat = pd.Series(tradeMat,index=tradeDates)
print 'Trade data loaded.'

#分层日度收益 L-Long S-Short
rateMatL = pd.DataFrame([[float('Nan')]*len(factors)]*len(tradeDates),columns=map(lambda x: x+' L', factors),index=tradeDates)
rateMatS = pd.DataFrame([[float('Nan')]*len(factors)]*len(tradeDates),columns=map(lambda x: x+' S', factors),index=tradeDates)
standard = pd.Series([float('Nan')]*len(tradeDates),index=tradeDates,name='standard')
for factor in factors:
    for i in range(afterwards[factor],len(d90)-2):
        codegroupL = datas[factor][i][ datas[factor][i] > np.percentile(datas[factor][i],80) ].index
        codegroupS = datas[factor][i][ datas[factor][i] <= np.percentile(datas[factor][i],20) ].index
        if factor in delayfactor:
            dates = tradeDates[(tradeDates>(d90[i+1]+delaytime[i%4]))&(tradeDates<=(d90[i+2]+delaytime[(i+1)%4]))]
        else:
            dates = tradeDates[(tradeDates>d90[i+1])&(tradeDates<=d90[i+2])]
        assetL = pd.Series([float(1)]*len(codegroupL),index=codegroupL)
        assetS = pd.Series([float(1)]*len(codegroupS),index=codegroupS)
        for date in dates:
            asset2L = (assetL*tradeMat[date]).dropna()
            asset2S = (assetS*tradeMat[date]).dropna()
            rateMatL.ix[date,factor+' L'] = float(sum(asset2L))/sum(assetL)
            rateMatS.ix[date,factor+' S'] = float(sum(asset2S))/sum(assetS)
            assetL = asset2L
            assetS = asset2S
        print i
    print factor
for i in range(0,len(d90)-2):
    dates = tradeDates[(tradeDates>d90[i+1])&(tradeDates<=d90[i+2])]
    asset = pd.Series([float(1)]*len(datas[0][i].index),index=datas[0][i].index)
    for date in dates:
        asset2 = (asset*tradeMat[date]).dropna()
        standard[date] = float(sum(asset2))/sum(asset)
        asset = asset2
    print i
print "Layers' yield calculated."

rateMat = pd.concat([standard,rateMatL,rateMatS],axis=1)
rateMat.to_csv(path2+'layerYield1d.csv',header=True)
