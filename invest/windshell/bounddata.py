#coding=utf8


import string
import pandas as pd
import numpy as np


df6 = pd.read_csv('./wind/bond_fund.CSV', index_col = 0, parse_dates = 'date')
df7 = pd.read_csv('./wind/bond_index.CSV', index_col = 0, parse_dates = 'date')
df8 = pd.read_csv('./wind/fund_value.csv', index_col = 0, parse_dates = 'date')


df = pd.concat([df6, df7, df8], axis = 1)
#print df
#df = df.fillna(method='pad')
#df = pd.concat([df, df8], axis = 1)


df = df.fillna(method='pad')

df = df.loc[df8.index]

df.to_csv('./wind/bond_value.csv')

#print df


'''
f = open('./wind/bounddata.csv', 'r')
lines = f.readlines()
for line in lines:
        line = line.strip()
        line = line.replace('-','')
        line = line.replace('\'','')
        print line

f.close()
'''
