# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import numpy as np
import os,sys

locpath = sys.path[0]

#生成至今为止的时间节点序列 #半年期
d90 = []
for year in range(2011,datetime.datetime.now().year):
    for month in range(0,2):
        d90 += [datetime.datetime(year,month*6+1,1)]
for month in range(0,(datetime.datetime.now().month-1)/6+1):
    d90 += [datetime.datetime(datetime.datetime.now().year,month*6+1,1)]
d90 = map(lambda x: x-datetime.timedelta(days=1),d90)

def daystr(datetime):
    return str(datetime)[0:4]+str(datetime)[5:7]+str(datetime)[8:10]

def datereplace(date):
    '''
    if '0630' in daystr(date):
        return datetime.datetime(date.year,8,31)
    elif '1231' in daystr(date):
        return datetime.datetime(date.year+1,3,31)
    '''
    return date

fundpath = u'/home/huyang/MultiFactors201710/new_funddata/'
exposepath = u'/home/huyang/MultiFactors201710/new_funddata/skdetail/'
valuepath = u'%s/cleanedData_standarded/' %locpath
outpath = u'%s/fundFactorValue/' %locpath

fsymbol = pd.read_csv(fundpath+'tq_fd_basicinfo.csv')
fsymbol = fsymbol.drop(fsymbol.columns[0],axis=1).set_index('SECODE')
ssymbol = pd.read_csv(fundpath+'tq_sk_basicinfo.csv')
ssymbol = ssymbol.drop(ssymbol.columns[0],axis=1).set_index('SECODE')

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

reportFactor = ['roe_q','roa_q','grossprofitmargin_q','assetturnover_q',
                'cashratio','currentratio','holder_avgpct','holder_avgpctchange_quart',
                'holder_avgpctchange_half','roe_ttm','roa_ttm','grossprofitmargin_ttm',
                'assetturnover_ttm','sales_growth_q','profit_growth_q',
                'operationcashflow_growth_q','sales_growth_ttm','profit_growth_ttm',
                'operationcashflow_growth_ttm','sales_growth_3y','profit_growth_3y',
                'operationcashflow_growth_3y','BP','SP','NCFP','OCFP','FCFP','DP',
                'EP','EPcut','profitmargin_q','operationcashflowradio_q',
                'financial_leverage','debtequityratio','profitmargin_ttm',
                'operationcashflowradio_ttm','marketvalue_leverage']

for factor in factors:
    matF = []
    for date in d90[1:(len(d90))]:
        try:
            if factor in reportFactor:
                rdate = datereplace(date)
            else:
                rdate = date
            value = pd.read_csv(valuepath+factor+'/'+daystr(rdate)+'.csv')
            value = pd.Series(list(value.iloc[:,1]),index=list(value.iloc[:,0]))
            expose = pd.read_csv(exposepath+daystr(date)+'.csv').set_index('SECODE').dropna()
            values = pd.Series(list(value[ssymbol.ix[expose.ix[:,'SKCODE'],0]]),index=expose.index)
            scores = values * expose.ix[:,'NAVRTO'] / 100
            line = scores.groupby(scores.index).sum().dropna().rename(daystr(datereplace(date)))
            matF += [line]
            print date
        except:
            pass
    matF = pd.concat(matF,axis=1)
    matF.index = fsymbol.ix[matF.index,0]
    matF = matF.groupby(matF.index).mean()
    matF.to_csv(outpath+factor+'.csv',header=True)
    print factor
