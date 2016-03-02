#coding=utf8



import string
import pandas as pd
import numpy as np



df1 = pd.read_csv('./wind/股票型基金.CSV',index_col = 0, parse_dates = '[0]')
df2 = pd.read_csv('./wind/偏股混合型基金.CSV',index_col = 0, parse_dates = '[0]')
df3 = pd.read_csv('./wind/平衡混合型基金.CSV',index_col = 0, parse_dates = '[0]')
df4 = pd.read_csv('./wind/灵活配置型.CSV',index_col = 0, parse_dates = '[0]')
df5 = pd.read_csv('./wind/指数数据.CSV',index_col = 0, parse_dates = '[0]')


df = pd.concat([df1, df2, df3, df4, df5], axis = 1, join_axes=[df5.index])


print df
print df5

df.to_csv('./wind/data.csv')
