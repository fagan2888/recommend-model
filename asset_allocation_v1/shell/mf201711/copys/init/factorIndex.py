#coding=utf-8
import pandas as pd
import numpy as np
import datetime
import os,sys
import statsmodels.formula.api as smf
import scipy.stats as stats

locpath = sys.path[0]

try:
    groupnum = int(sys.argv[1])
except:
    groupnum = 5

#生成至今为止的时间节点序列
d90 = []
for year in range(2004,datetime.datetime.now().year):
    for month in range(0,12):
        d90 += [datetime.datetime(year,month+1,1)]
for month in range(0,4):
    d90 += [datetime.datetime(datetime.datetime.now().year,month+1,1)]
d90 = d90[3:len(d90)]
d90 = map(lambda x: x-datetime.timedelta(days=1),d90)

def daystr(datetime):
    return str(datetime)[0:4]+str(datetime)[5:7]+str(datetime)[8:10]

#环境变量，因子名称赋值
datepath = '/home/huyang/MultiFactors201710/originalData/TQ_SK_YIELDINDIC/'
datapath = '%s/cleanedData_standarded/' %locpath

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
afterwards = pd.Series([0,0,0,0,0,0,0,0,0,12,12,12,
                        12,4,4,4,12,12,12,36,36,
                        36,0,0,0,0,0,12,12,4,0,66,66,12,
                        12,0,0,0,2,8,0,0,0,
                        0,0,0,0,0,0,0,
                        0,0,2,8,0,0,0,0,0,0,2,8],index=factors)

#因子数据读取
datas = []
for factor in factors:
    dataone = [[]]*afterwards[factor]
    for i in range(afterwards[factor],len(d90)-2):
        data = pd.read_csv(datapath+factor+'/'+daystr(d90[i+1])+'.csv')
        datatmp = data[data.columns[1]].apply(lambda x: float(x))
        datatmp.index = data[data.columns[0]].apply(lambda x: str(x))
        dataone += [datatmp]
    datas += [dataone]
datas = pd.Series(datas,index=factors)
print 'Factor data loaded.'

#行情数据读取
tradeDates = pd.Series(os.listdir(datepath)).apply(lambda x: datetime.datetime.strptime(str(x).replace('.csv',''),'%Y%m%d'))
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
        codegroupL = datas[factor][i][ datas[factor][i] > np.percentile(datas[factor][i],100.0-100.0/groupnum) ].index
        codegroupS = datas[factor][i][ datas[factor][i] <= np.percentile(datas[factor][i],100.0/groupnum) ].index
        dates = tradeDates[(tradeDates>d90[i+1])&(tradeDates<=d90[i+2])]
        assetL = pd.Series([float(1)]*len(codegroupL),index=codegroupL)
        assetS = pd.Series([float(1)]*len(codegroupS),index=codegroupS)
        for date in dates:
            asset2L = (assetL*(tradeMat[date][assetL.index].fillna(1))).dropna()
            asset2S = (assetS*(tradeMat[date][assetS.index].fillna(1))).dropna()
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
rateMat.to_csv('%s/layerYield1d.csv' %locpath ,header=True)
