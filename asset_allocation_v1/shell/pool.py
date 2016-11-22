#coding=utf8

import pandas as pd

df = pd.read_csv('aa.csv', index_col = 'code')

print df.iloc[0]
