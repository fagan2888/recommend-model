#coding=utf-8
import pandas as pd
import numpy as np
import os,sys

locpath = sys.path[0]

rpath = '/home/huyang/MultiFactors201710/cleanedData/'
wpath = '%s/cleanedData_standarded/' %locpath
validpath = '%s/validcode/' %locpath

folders = pd.Series(os.listdir(rpath))
#folders = pd.Series(['tradevolumn_1m','tradevolumn_3m','tradevolumn_6m','tradevolumn_12m'])
#folders = pd.Series(['roe_q','roe_ttm','cashratio'])

try:
    os.makedirs(wpath+'rate/')
except:
    pass

for folder in folders:
    try:
        os.makedirs(wpath+folder+'/')
    except:
        pass
    files = os.listdir(rpath+folder+'/')
    for fileone in files:
        data = pd.read_csv(rpath+folder+'/'+fileone)
        series = data.set_index(data.columns[0]).iloc[:,0]
        #先取有效股票的数据集再做标准化
        data = pd.read_csv(validpath+fileone)
        validcode = data.iloc[:,1]
        series = series[list(set(series.index).intersection(set(validcode)))]
        #去除缺失值与无意义值
        series = series[(series!=float('Inf'))&(series!=float('-Inf'))].dropna()
        #复制一个因变量rate（三月收益率），不作标准化
        if folder == 'relative_strength_1m':
            series.to_csv(wpath+'rate/'+fileone,header=True)
        #去极值与标准化
        if len(series)>0:            
            sd = np.std(series)+1e-16
            median = np.percentile(series,50)
            series[series > median + 5*sd] = median + 5*sd
            series[series < median - 5*sd] = median - 5*sd
            sdnew = np.std(series)+1e-16
            meannew = np.mean(series)
            series = (series-meannew)/sdnew
        series.to_csv(wpath+folder+'/'+fileone,header=True)
        print folder, fileone
