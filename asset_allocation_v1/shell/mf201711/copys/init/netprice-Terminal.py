#coding=utf-8
import pandas as pd
import numpy as np
import datetime
import os,sys

locpath = sys.path[0]

#生成至今为止的时间节点序列 #每年times次
times = 12
'''
d90 = []
for year in range(2012,datetime.datetime.now().year):
    for month in range(0,times):
        d90 += [datetime.datetime(year,month*(12/times)+1,1)]
for month in range(0,4):
    d90 += [datetime.datetime(datetime.datetime.now().year,month*(12/times)+1,1)]
d90 = map(lambda x: x-datetime.timedelta(days=1),d90)
'''
def daystr(datetime):
    return str(datetime)[0:4]+str(datetime)[5:7]+str(datetime)[8:10]

#环境变量
kinds = 5
inimonth = 8
vision = ''
folder = u'-Terminal'
fundpath = u'/home/huyang/MultiFactors201710/new_funddata/price1d/'
poolpath = locpath+u'/fundPool'+folder+'/pool'+str(kinds)+'-'
outpath = locpath+'/netprice'+folder+'/'
factorpath = '%s/layerYield1d.csv' %locpath

factors = ['roe_q','roa_q','grossprofitmargin_q','assetturnover_q',
               'cashratio','currentratio','holder_avgpct','holder_avgpctchange_quart',
               'holder_avgpctchange_half','roe_ttm','roa_ttm','grossprofitmargin_ttm',
               'assetturnover_ttm','sales_growth_q','profit_growth_q',
               'operationcashflow_growth_q','sales_growth_ttm','profit_growth_ttm',
               'operationcashflow_growth_ttm','sales_growth_3y','profit_growth_3y',
               'operationcashflow_growth_3y','BP','SP','NCFP','OCFP','FCFP',
               'EP','EPcut','profitmargin_q','operationcashflowradio_q',
               'financial_leverage','debtequityratio','profitmargin_ttm',
               'operationcashflowradio_ttm','marketvalue_leverage',
               'high_low_1m','high_low_3m','high_low_6m','high_low_12m',
               'ln_capital','ln_price','tradevolumn_1m',
               'tradevolumn_3m','tradevolumn_6m','tradevolumn_12m',
               'relative_strength_1m','relative_strength_3m',
               'relative_strength_6m','relative_strength_12m',
               'std_1m','std_3m','std_6m','std_12m',
               'turnover_1m','turnover_3m','turnover_6m','turnover_12m',
               'weighted_strength_1m','weighted_strength_3m',
               'weighted_strength_6m','weighted_strength_12m']
factors = map(lambda x: x+' L', factors) + map(lambda x: x+' S', factors)

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

#def netp():
#因子指数读取
indexes = pd.read_csv(factorpath)
indexes = indexes.set_index(indexes.columns[0])
try:
    indexes.index = map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m-%d'), indexes.index)
except:
    indexes.index = map(lambda x: datetime.datetime.strptime(str(x),'%Y/%m/%d'), indexes.index)
print 'Factor index data loaded.'
#factors = list(indexes.columns[1:indexes.shape[1]])

#基金池净值及数据指标计算
characterMat = pd.DataFrame([[float('nan')]*5]*(indexes.shape[1]-1),index=indexes.columns[1:indexes.shape[1]],columns=['turnover','sharpratio','annualyield','annualvolatility','meanValue'])
for factor in factors:
    pool = pd.read_csv(poolpath+factor+'.csv')#.drop(range(0,times/2))
    pool = pool.set_index(pool.columns[0])
    try:
        pool.index = map(lambda x: datetime.datetime.strptime(str(x),'%Y-%m-%d'), pool.index)
    except:
        try:
            pool.index = map(lambda x: datetime.datetime.strptime(str(x),'%Y/%m/%d'), pool.index)
        except:
            pool.index = map(lambda x: datetime.datetime.strptime(str(x),'%Y%m%d'), pool.index)
    accumulate = 1
    netprices = []
    d90 = list(pool.index[0:len(pool.index)])
    for date in range(inimonth,len(d90)-1):
        funds = pool.ix[d90[date],map(lambda x: 'fund'+str(x+1), range(0,kinds))].dropna().apply(lambda x: str(int(x)))
        ratetmp = rates.ix[(rates.index>=d90[date])&(rates.index<=d90[date+1]),funds].fillna(method='pad',axis=0)
        netprice = ratetmp.div(ratetmp.iloc[0,:],axis=1).mean(axis=1)*accumulate
        accumulate = netprice[d90[date+1]]
        netprices += [netprice[1:len(netprice)]]
    netprices = pd.concat(netprices,axis=0)
    #outMat = pd.concat([netprices,indexes.ix[netprices.index,factor].cumprod()],axis=1).dropna()
    #outMat = outMat.div(outMat.iloc[0,:],axis=1)
    #outMat.columns = ['fundPoolNetPrice','factorIndex']
    #outMat.to_csv(outpath+factor+vision+'.csv',header=True)
    netprices.to_csv(outpath+factor+vision+'.csv',header=True)
    characterMat.ix[factor,'turnover'] = np.mean(pool.ix[1:pool.shape[0],'turnover'])
    characterMat.ix[factor,'annualyield'] = pow(accumulate,float(times)/(pool.shape[0]-inimonth)) - 1
    dailyyields = (netprices/netprices.shift(1)).dropna() - 1
    characterMat.ix[factor,'annualvolatility'] = np.std(dailyyields[dailyyields!=0]) * np.sqrt(250)
    characterMat.ix[factor,'sharpratio'] = (characterMat.ix[factor,'annualyield']-0.03)/characterMat.ix[factor,'annualvolatility']
    characterMat.ix[factor,'meanValue'] = np.mean(pool.ix[:,'meanValue'])
    print factor
#characterMat.to_csv(outpath+'summary'+vision+'.csv',header=True)
print 'Fund pool net price calculated.'
