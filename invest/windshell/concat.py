#coding=utf8

import pandas as pd


df1 = pd.read_csv('./tmp/indicator.csv', index_col = 'code')

df2 = pd.read_csv('./tmp/tags.csv', index_col = 'code')


df = pd.concat([df1, df2], axis = 1, join_axes=[df2.index])


df.to_csv('./tmp/indicator_tags.csv')
