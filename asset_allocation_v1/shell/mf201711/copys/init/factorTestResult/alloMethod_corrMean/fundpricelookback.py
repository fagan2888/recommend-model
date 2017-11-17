#coding=utf-8
import pandas as pd
import numpy as np
import os,sys
import datetime

dirpath = sys.path[0]+'/'

poolpath = dirpath+u'../../fundPool-Terminal/'
netpath = dirpath+u'../../netprice-Terminal/'
grouppath = dirpath+u'../groupCode/'

allo = pd.read_csv(dirpath+'alloDirMat.csv',index_col=0,parse_dates=[0])
alloL = allo[allo==1]
alloS = -allo[allo==-1]
alloL.columns = map(lambda x: x+' L' ,alloL.columns)
alloS.columns = map(lambda x: x+' S' ,alloS.columns)
alloMat = pd.concat([alloL,alloS],axis=1)
netlist = os.listdir(netpath)
rates = []
for factor in netlist:
    data = pd.read_csv(netpath+factor,index_col=0,parse_dates=[0])
    data.columns = [factor.replace('.csv','')]
    rates += [data/data.shift(1)]
    print factor
rates = pd.concat(rates,axis=1)

#turnover是算给定因子下的单因子基金池平均换手回测
factors = map(lambda x: x.replace('.csv',''), os.listdir(poolpath))
datas = []
for factor in factors:
    data = pd.read_csv(poolpath+factor+'.csv')
    data = data.set_index(data.columns[0])
    datas += [data.ix[:,'turnover']]
datas = pd.concat(datas,axis=1)
datas.columns = factors
datas = datas.dropna(how='all')
datas.ix['annualturn',:] = datas.mean()*12
datas.to_csv(dirpath+'turnoverConcat.csv',header=True)
Fundpoolturn = datas.ix['annualturn',:].mean()

#backMat是算给定因子下的基金净值回测
backMat = pd.DataFrame(columns=rates.columns)
for date in rates.index:
    backdate = datetime.datetime(date.year,date.month,1) - datetime.timedelta(days=1)
    backMat.ix[date,:] = rates.ix[date,:].multiply(alloMat.ix[backdate,:])
    print date
backrate = backMat.mean(axis=1).fillna(1)
backprice = backrate.cumprod()
price = backrate.prod()
annual = np.power(price,365.0/len(backrate)) - 1
sigma = np.std(backrate) * np.sqrt(365)
sharp = (annual-0.03) / sigma
backMat.ix[:,'market'] = rates.mean(axis=1)
backMat.ix[:,'rate'] = backrate
backMat.ix['factorcontribute',:] = backMat.div(rates.mean(axis=1),axis=0).prod(axis=0)
backMat.ix[:,'price'] = backprice
backMat.to_csv(dirpath+'backMat.csv')

#position是算给定因子下的股票换手率回测
position = pd.DataFrame()
for date in alloMat.index:
    postmp = pd.DataFrame()
    for factor in alloMat.ix[date,:].dropna().index:
        data = pd.read_csv(grouppath+factor+'/'+datetime.datetime.strftime(date,'%Y%m%d')+'.csv',header=None,index_col=1)
        postmp = postmp.append((data>-1).T/len(data)/len(alloMat.ix[date,:].dropna()))
    position = position.append(postmp.sum().rename(date))
    print date
turnrate = abs(position.fillna(0) - position.fillna(0).shift(1)).sum(axis=1)/2
turnrate.rename('turnrate').to_csv(dirpath+'stockturnMat.csv')
annualturn = turnrate.dropna().mean() * 12

#position又是算给定因子下的基金换手率回测
position = pd.DataFrame()
for date in alloMat.index:
    try:
        postmp = pd.DataFrame()
        for factor in alloMat.ix[date,:].dropna().index:
            data = pd.read_csv(poolpath+'pool2-'+factor+'.csv',index_col=0,parse_dates=[0])
            data = data.ix[date,pd.Series(map(lambda x: 'fund' in x, data.columns),index=data.columns)]
            data = pd.DataFrame([[1.0]*len(data)],columns=map(lambda x: str(x).replace('.0',''), data.values))
            postmp = postmp.append((data>-1)/data.shape[1]/len(alloMat.ix[date,:].dropna()))
        position = position.append(postmp.sum().rename(date))
        print date
    except:
        pass
turnrate = abs(position.fillna(0) - position.fillna(0).shift(1)).sum(axis=1)/2
turnrate.rename('turnrate').to_csv(dirpath+'fundpoolturnMat.csv')
fpannualturn = turnrate.dropna().mean() * 12

result = pd.Series([annualturn,annual,sigma,sharp,Fundpoolturn,fpannualturn],
                   index=['Stockannualturn','Fundyield','Fundsigma','Fundsharp',
                          'AllFundpoolturn','UsedFundpoolturn'])
print list(result)
result.to_csv(dirpath+'fundresult.csv')
