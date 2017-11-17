#coding=utf-8
import pandas as pd
import numpy as np
import os,sys
import time

dirpath = sys.path[0]+'/'
path = u'../'
layer = pd.read_csv(dirpath+path+'layeredSummary.csv',index_col=0,parse_dates=[0])
grate = pd.read_csv(dirpath+path+'GroupRate.csv',index_col=0,parse_dates=[0])
g1rate = pd.read_csv(dirpath+path+'G1Rate.csv',index_col=0,parse_dates=[0])
g5rate = pd.read_csv(dirpath+path+'G5Rate.csv',index_col=0,parse_dates=[0])
TV = []#pd.read_csv(dirpath+path+'TValue.csv',index_col=0,parse_dates=[0])
RV = []#pd.read_csv(u'C:/Users/Administrator/Desktop/多因子每月2017.10/fundFactorValueRepresent2.csv',index_col=0,parse_dates=[0])

#因子分类
ValueFactor = ['EP','EPcut','BP','SP','NCFP','OCFP']
GrowthFactor = ['sales_growth_q','profit_growth_q','operationcashflow_growth_q',
                'sales_growth_ttm','profit_growth_ttm','operationcashflow_growth_ttm',
                'sales_growth_3y','profit_growth_3y','operationcashflow_growth_3y']
FinancialFactor = ['roe_q','roa_q','roe_ttm','roa_ttm','grossprofitmargin_q',
                   'grossprofitmargin_ttm','profitmargin_q','profitmargin_ttm',
                   'assetturnover_q','assetturnover_ttm',
                   'operationcashflowradio_q','operationcashflowradio_ttm']
LeverageFactor = ['marketvalue_leverage','financial_leverage','debtequityratio',
                  'cashratio','currentratio']
SizeFactor = ['ln_capital']
MomentumFactor = ['relative_strength_1m','relative_strength_3m',
                  'relative_strength_6m','relative_strength_12m',
                  'weighted_strength_1m','weighted_strength_3m',
                  'weighted_strength_6m','weighted_strength_12m']
VolatilityFactor = ['high_low_1m','high_low_3m','high_low_6m','high_low_12m',
                    'std_1m','std_3m','std_6m','std_12m','ln_price']
TurnoverFactor = ['turnover_1m','turnover_3m','turnover_6m','turnover_12m']
ShareholderFactor = ['holder_avgpct','holder_avgpctchange_quart','holder_avgpctchange_half']
TradevolumnFactor = ['tradevolumn_1m','tradevolumn_3m','tradevolumn_6m','tradevolumn_12m']

factorbase = [ValueFactor,GrowthFactor,FinancialFactor,LeverageFactor,SizeFactor,
              MomentumFactor,VolatilityFactor,TurnoverFactor,ShareholderFactor,TradevolumnFactor]
factorkind = ['Value','Growth','Financial','Leverage','Size',
              'Momentum','Volatility','Turnover','Shareholder','Tradevolumn']

tradeCost = 0.002 #股票一进一出交易费率按0.2%，基金按1%

def test(
    if_DynamicPeriods = False, maxPeriods = 24, minPeriods = 1, layerPeriod = 14,
    if_GRLimit = False, grPeriod = 24, grLimit = 30, grTop = 100,
    if_RVLimit = False, rvLimit = 1,
    if_BaseLimit = True, baserankLimit = 2,
    filter_Method = ['percentile','rank','value_percentile','value_rank','value'][0],
    layerLimit = [95,4][0], layerBottom = 0.5, layerTop = 1,
    is_Print = False, is_Cost = False, tradeCost = tradeCost):

    #动态期限或固定期限
    if if_DynamicPeriods:
        layermean = pd.DataFrame(index=layer.index,columns=layer.columns)
        tmpRates = []
        for Period in range(1,25):
            tmpRate = test(
                if_DynamicPeriods = False, layerPeriod = Period,
                if_GRLimit = if_GRLimit, grPeriod = grPeriod, grLimit = grLimit,
                if_BaseLimit = if_BaseLimit, baserankLimit = baserankLimit,
                filter_Method = filter_Method, layerLimit = layerLimit,
                layerBottom = layerBottom, layerTop = layerTop, is_Print = False)
            tmpRates += [tmpRate[0]]
            print Period,tmpRate[1:4]
        tmpRates = pd.concat(tmpRates,axis=1)
        tmpRates.columns = range(1,25)
        tmpRateMean = (tmpRates.shift(-1,axis=1) + tmpRates + tmpRates.shift(1,axis=1)) #是否采用相邻期平滑算法
        #tmpRateMean = tmpRates #是否采用相邻期平滑算法
        layerPeriod = tmpRateMean.idxmax(axis=1).shift(1).dropna()
        #return tmpRateMean,layerPeriod
        tmpRates.ix[:,'choose'] = layerPeriod
        tmpRates.to_csv(dirpath+'mat_periodrate2.csv')
        for date in layerPeriod.index:
            layermean.ix[date,:] = layer.rolling(int(layerPeriod[date])).mean().shift(1).ix[date,:]
        layermean = layermean.dropna(axis=0,how='all')

    else:
        layermean = layer.rolling(layerPeriod).mean().shift(1).dropna(axis=0,how='all')
        layermean = layermean.iloc[(24-layerPeriod):layermean.shape[0],:] #保证在同时期进行对比

    layervalidmeanabs = np.abs(layermean)
    lvma = layervalidmeanabs
    rate = g1rate*(layermean>0) + g5rate*(layermean<0)

    #收益率限定
    if if_GRLimit:
        grmean = np.abs(grate).rolling(grPeriod).mean().shift(1).dropna(axis=0,how='all')
        lvma[grmean.lt(grmean.apply(lambda x: np.percentile(x,grLimit),axis=1),axis=0)] = 0
        lvma[grmean.gt(grmean.apply(lambda x: np.percentile(x,grTop),axis=1),axis=0)] = 0

    #“因子的基金代表性”限定
    if if_RVLimit:
        lvma = lvma.ix[lvma.index>=RV.index[0],:]
        for date in lvma.index:
            rdate = max(RV.index[RV.index<=date])
            for factor in lvma.columns:
                if np.sign(layermean.ix[date,factor]) == 1 and RV.ix[rdate,factor+' L'] < rvLimit:
                    lvma.ix[date,factor] = 0
                    #print date,rdate,factor+' L'
                elif np.sign(layermean.ix[date,factor]) == -1 and RV.ix[rdate,factor+' S'] < rvLimit:
                    lvma.ix[date,factor] = 0
                    #print date,rdate,factor+' S'
    
    #大类内部因子数限定
    if if_BaseLimit:
        for date in lvma.index:
            for base in factorbase:
                lvma.ix[date,lvma.ix[date,base].dropna().sort_values(ascending=False)[baserankLimit:len(base)].index] = 0
    
    #分层分数筛选算法
    alloMat = pd.DataFrame(index=lvma.index,columns=lvma.columns)
    if 'percentile' in filter_Method:
        lvmaLimit = lvma.apply(lambda x: np.percentile(x.dropna(),layerLimit),axis=1) #百分位筛选
        alloMat[lvma.ge(lvmaLimit,axis=0)] = 1
    if 'rank' in filter_Method:
        lvmaLimit = lvma.apply(lambda x: x.sort_values(ascending=False)[layerLimit-1],axis=1) #排序筛选
        alloMat[lvma.ge(lvmaLimit,axis=0)] = 1
    if 'value' in filter_Method:
        alloMat[(lvma>layerBottom)&(lvma<=layerTop)] = 1

    #回测
    excessMat = (alloMat * grate * layermean.apply(lambda x: x.apply(lambda y: np.sign(y))) / 2).ix[alloMat.index,:] #不知为什么np.sign不能直接用了
    excessArray = excessMat.mean(axis=1).fillna(0) #超额收益率
    excesspriceArray = (excessArray+1).cumprod() #超额收益倍数

    rateMat = (alloMat * rate).ix[alloMat.index,:]
    rateArray = rateMat.mean(axis=1).fillna(0) #收益率
    priceArray = (rateArray+1).cumprod() #净值

    turnMat = (alloMat.div(alloMat.sum(axis=1),axis=0)).fillna(0)
    turnArray = np.abs((turnMat - turnMat.shift(1)).dropna()).sum(axis=1)/2
    annualturn = turnArray.mean()*12

    if is_Cost:
        rateArray = rateArray - turnArray * tradeCost #带交易费率收益率
        priceArray = (rateArray+1).cumprod() #带交易费率净值

    meanfactornum = alloMat.sum(axis=1).mean()

    annual = np.power(priceArray[len(priceArray)-1],float(12)/len(priceArray)) - 1
    sigma = np.std(rateArray) * np.sqrt(12)
    sharpe = (annual-0.03) / sigma

    #输出配置表
    if is_Print:
        alloDirMat = alloMat * layermean.apply(lambda x: x.apply(lambda y: np.sign(y)))
        alloDirMat.to_csv(dirpath+'alloDirMat.csv')
        alloMat.ix[:,'totalnum'] = alloMat.sum(axis=1).fillna(0)
        for i in range(0,len(factorkind)):
            alloMat.ix[:,factorkind[i]] = alloMat.ix[:,factorbase[i]].sum(axis=1)
        alloMat.ix[:,'price'] = priceArray
        alloMat.ix[:,'excess'] = excesspriceArray
        alloMat.ix[:,'market'] = priceArray.div(excesspriceArray)
        alloMat.to_csv(dirpath+'alloMat.csv')

    #print round(sharpe,3),round(annualturn,2),round(meanfactornum,1)
    return excessArray,round(sharpe,3),round(annualturn,2),round(meanfactornum,1)

result = pd.Series(test(is_Print=True)[1:4],index=['Stocksharpe','Factorannualturn','meanfactornum'])
print list(result)
result.to_csv(dirpath+'stockresult.csv')
