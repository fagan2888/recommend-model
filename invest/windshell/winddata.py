#coding=utf8

import string
import pandas as pd
import numpy as np

df1 = pd.read_csv('./wind/股票型基金.CSV',index_col = 0, parse_dates = 'date')
df2 = pd.read_csv('./wind/偏股混合型基金.CSV',index_col = 0, parse_dates = 'date')
df3 = pd.read_csv('./wind/平衡混合型基金.CSV',index_col = 0, parse_dates = 'date')
df4 = pd.read_csv('./wind/灵活配置型基金.CSV',index_col = 0, parse_dates = 'date')
df5 = pd.read_csv('./wind/指数数据.CSV',index_col = 0, parse_dates = 'date')

#df = pd.concat([df1, df2, df3, df4, df5], axis = 1, join_axes=[df5.index])
df = pd.concat([df1, df2, df3, df4, df5], axis = 1)


#print df5.index.values

#print df
#print df5


#print df.index.values
#dates = df.index.values
#dates.sort()
#print dates
#df.reindex(dates)


#print df

'''
for col in df.columns:
    values = df[col]
    for i in range(1, len(values)):
        if np.isnan(values[i]):
            values[i] = values[i -1]        
'''
#print df
df = df.fillna(method='pad')
#df.fillna(method='pad')

#df['163001.OF'].to_csv('./tmp/163001.csv')
df.to_csv('./wind/data.csv')


f = open('./wind/data.csv', 'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.replace('-','')
    line = line.replace('\'','')
    #print line

f.close()
