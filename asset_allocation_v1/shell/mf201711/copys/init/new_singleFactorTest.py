#coding=utf8

import pandas as pd
import datetime
import os,sys
import math
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
import scipy.stats as stats

locpath = sys.path[0]

try:
    groupnum = int(sys.argv[1])
except:
    groupnum = 5

#生成至今为止的时间节点序列 2004.4.30 - 2017.3.31
d90 = []
for year in range(2004,datetime.datetime.now().year):
    for month in range(0,12):
        d90 += [datetime.datetime(year,month+1,1)]
for month in range(0,4):
#for month in range(0,10):
    d90 += [datetime.datetime(datetime.datetime.now().year,month+1,1)]
d90 = d90[4:len(d90)]
d90 = map(lambda x: x-datetime.timedelta(days=1),d90)

def daystr(datetime):
    return str(datetime)[0:4]+str(datetime)[5:7]+str(datetime)[8:10]

#环境变量，记录各类数据的导入导出路径
rpath = '%s/cleanedData_standarded/' %locpath
wpath = '%s/factorTestResult/' %locpath
indexpath = wpath + 'layeredFactorIndex/' #因子指数输出路径
codepath = wpath + 'groupCode/' #股票池输出路径
try:
    os.makedirs(wpath)
except:
    pass

#固定的因变量数据读取
rate = []
for i in range(2,len(d90)):
    data = pd.read_csv(rpath+'rate/'+daystr(d90[i])+'.csv')
    ratetmp = data[data.columns[1]].apply(lambda x: float(str(x)))/100
    ratetmp.index = data[data.columns[0]].apply(lambda x: str(x))
    rate += [ratetmp]
#'''
#因子名称及缺失期数赋值
factornames = ['roe_q','roa_q','grossprofitmargin_q','assetturnover_q',
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
                        0,0,2,8,0,0,0,0,0,0,2,8],index=factornames)

#数据读取
print 'Single factor test is started.'
datas = []
for factor in factornames:
    dataone = [[]]*afterwards[factor]
    for i in range(afterwards[factor],len(d90)-2):
        data = pd.read_csv(rpath+factor+'/'+daystr(d90[i+1])+'.csv')
        datatmp = data[data.columns[1]].apply(lambda x: float(x))
        datatmp.index = data[data.columns[0]].apply(lambda x: str(x))
        dataone += [datatmp]
    datas += [dataone]
datas = pd.Series(datas,index=factornames)
print 'Data loaded.'

#分层秩相关评测
try:
    os.makedirs(indexpath)
except:
    pass

for factor in factornames:
    groupMat = pd.DataFrame([[float('Nan')]*(2+groupnum)]*(len(d90)-2-afterwards[factor]),
                            index = map(lambda x: daystr(x), d90[(2+afterwards[factor]):len(d90)]),
                            columns = map(lambda x: 'Group'+str(x), range(1,groupnum+1))+['Rank','SpearmanCorr'])
    try:
        os.makedirs(codepath+factor+' L/')
        os.makedirs(codepath+factor+' S/')
    except:
        pass
    for i in range(afterwards[factor],len(d90)-2):
        gcodes = []
        for gi in range(0,groupnum):
            gcodes += [datas[factor][i][(datas[factor][i] > np.percentile(datas[factor][i],100.0-100.0/groupnum*(gi+1))) & (datas[factor][i] <= np.percentile(datas[factor][i],100.0-100.0/groupnum*gi))].index]
        pd.Series(gcodes[0]).to_csv(codepath+factor+' L/'+daystr(d90[i+2])+'.csv')
        pd.Series(gcodes[groupnum-1]).to_csv(codepath+factor+' S/'+daystr(d90[i+2])+'.csv')
        grouprates = map(lambda x: np.mean(rate[i][list(set(x).intersection(set(rate[i].index)))].dropna()), gcodes)
        groupMat.ix[daystr(d90[i+2])] = grouprates + ['.'.join(map(lambda x:str(x+1), np.argsort(map(lambda y: -y, grouprates))))] + [stats.stats.spearmanr(pd.Series(grouprates).dropna(),range(len(pd.Series(grouprates).dropna()),0,-1))[0]]
    groupMat.to_csv(indexpath+factor+'.csv',header=True)
    print factor
print "Layers' spearman-test ended."
