#coding=utf-8
import pandas as pd
import numpy as np
import datetime
import sys
from globalvalue import *

path = '4_factorTiming/'
price = pd.read_csv(path+'fundNetprice.csv')
price = price.set_index(price.columns[0])
try:
    price.index = map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'), price.index)
except:
    price.index = map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'), price.index)

factor = ['ln_capital','BP','std_3m','tradevolumn_3m','holder_avgpct']

rate = (price/price.shift(1)-1).dropna()
rate = rate.ix[rate.iloc[:,0]!=0,:]
price = price.ix[rate.index,:]

shyrperatios = shyrperatios_42
trendratios = trendratios_42
periods = periods_42
limitpcts = limitpcts_42
directs = directs_42
trendstables = trendstables_42

for nameid in idrange:
    if nameid == 0:
        namestr = ''
    else:
        namestr = str(nameid)

    #shifter_algorithm
    shyrperatio = shyrperatios[nameid]
    shyrpe = ((np.power(price/price.shift(20),12)-1.03)/(rate.rolling(20).std()) - shyrperatio*(np.power(price/price.shift(60),4)-1.03)/(rate.rolling(60).std()))/np.sqrt(240)
    shyrpe = shyrpe.dropna()

    trendratio = trendratios[nameid]
    trend = pd.DataFrame([[float('Nan')]*len(factor)]*shyrpe.shape[0],index=shyrpe.index,columns=factor)
    for col in trend.columns:
        for row in range(0,trend.shape[0]):
            if (shyrpe.ix[trend.index[row],col+' L']-shyrpe.ix[trend.index[row],col+' S']) > (abs(shyrpe.ix[trend.index[row],col+' L'])+abs(shyrpe.ix[trend.index[row],col+' S']))*trendratio:
                trend.ix[trend.index[row],col] = 1
            elif (shyrpe.ix[trend.index[row],col+' S']-shyrpe.ix[trend.index[row],col+' L']) > (abs(shyrpe.ix[trend.index[row],col+' L'])+abs(shyrpe.ix[trend.index[row],col+' S']))*trendratio:
                trend.ix[trend.index[row],col] = -1
            else:
                if row == 0 or trendstables[nameid] == 0:
                    trend.ix[trend.index[row],col] = 0
                else:
                    trend.ix[trend.index[row],col] = trend.ix[trend.index[row-1],col]

    period = periods[nameid]
    limitpct = limitpcts[nameid]
    position = pd.DataFrame([[float('Nan')]*len(factor)]*trend.shape[0],index=trend.index,columns=factor)
    position.iloc[period-2,:] = (directs[nameid] == (trend.iloc[0:(period-1),:].sum(axis=0)>=0).apply(lambda x:int(x))).apply(lambda x:int(x))
    for row in range(period,position.shape[0]+1):
        for col in position.columns:
            if trend.ix[trend.index[(row-period):row],col].sum() >= limitpct:
                position.ix[position.index[row-1],col] = directs[nameid]
            elif trend.ix[trend.index[(row-period):row],col].sum() <= -limitpct:
                position.ix[position.index[row-1],col] = 1 - directs[nameid]
            else:
                position.ix[position.index[row-1],col] = position.ix[position.index[row-2],col]
    position = position.dropna()

    adjustday = (position != position.shift(1)).sum(axis=1)
    adjustday = adjustday[adjustday>0].index

    pos = pd.concat([position,1-position],axis=1).ix[adjustday,:]
    pos.columns = map(lambda x: x+' L', position.columns)+map(lambda x: x+' S', position.columns)

    weight = 1/rate.rolling(59).std().dropna().ix[adjustday,:]
    pos = (pos*weight).dropna()
    pos = pos.divide(pos.sum(axis=1),axis=0)

    pos.to_csv(path+'position'+namestr+'.csv',header=True)
