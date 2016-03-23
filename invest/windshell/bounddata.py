#coding=utf8


import string
import pandas as pd
import numpy as np


df6 = pd.read_csv('./wind/债券型基金.CSV', index_col = 0, parse_dates = '[0]')
df7 = pd.read_csv('./wind/债券指数.CSV', index_col = 0, parse_dates = '[0]')

df = pd.concat([df6, df7], axis = 1, join_axes=[df7.index])

df.to_csv('./wind/bounddata.csv')

#print df

f = open('./wind/bounddata.csv', 'r')
lines = f.readlines()
for line in lines:
        line = line.strip()
        line = line.replace('-','')
        line = line.replace('\'','')
        print line

f.close()
