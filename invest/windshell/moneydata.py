#coding=utf8

import string
import pandas as pd
import numpy as np

df6 = pd.read_csv('./wind/money.CSV', index_col = 0, parse_dates = 'date')
df8 = pd.read_csv('./wind/fund_value.csv', index_col = 0, parse_dates = 'date')

df = pd.concat([df6, df8], axis = 1)

#df = df.fillna(method='pad')
#df = pd.concat([df, df8], axis = 1)

df = df.fillna(method='pad')
df = df.loc[df8.index]

df.to_csv('./wind/money_value.csv')


