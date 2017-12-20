#coding=utf-8
import pandas as pd
import numpy as np
import os,sys
import time

locpath = sys.path[0]

dirpath = locpath+'/'
path = 'layeredFactorIndex/'
SCorrList = map(lambda x:x.replace('.csv',''),os.listdir(dirpath+path))

#'''
corrArrays = []
for csv in SCorrList:
    data = pd.read_csv(dirpath+path+csv+'.csv',index_col=0,parse_dates=[0])
    corrArray = data.ix[:,'SpearmanCorr']
    corrArray.name = csv
    corrArrays += [corrArray]
corrArrays = pd.concat(corrArrays,axis=1)
corrArrays.to_csv(dirpath+'layeredSummary.csv')
#'''

corrArrays = []
for csv in SCorrList:
    data = pd.read_csv(dirpath+path+csv+'.csv',index_col=0,parse_dates=[0])
    corrArray = data.ix[:,'Group1'] - data.ix[:,'Group'+str(data.shape[1]-2)]
    corrArray.name = csv
    corrArrays += [corrArray]
corrArrays = pd.concat(corrArrays,axis=1)
corrArrays.to_csv(dirpath+'GroupRate.csv')

corrArrays = []
for csv in SCorrList:
    data = pd.read_csv(dirpath+path+csv+'.csv',index_col=0,parse_dates=[0])
    corrArray = data.ix[:,'Group1']
    corrArray.name = csv
    corrArrays += [corrArray]
corrArrays = pd.concat(corrArrays,axis=1)
corrArrays.to_csv(dirpath+'G1Rate.csv')

corrArrays = []
for csv in SCorrList:
    data = pd.read_csv(dirpath+path+csv+'.csv',index_col=0,parse_dates=[0])
    corrArray = data.ix[:,'Group'+str(data.shape[1]-2)]
    corrArray.name = csv
    corrArrays += [corrArray]
corrArrays = pd.concat(corrArrays,axis=1)
corrArrays.to_csv(dirpath+'G5Rate.csv')
#'''
