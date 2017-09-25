#coding=utf8

import pandas as pd
import numpy as np

fund_df = pd.read_csv('./data/fund_nav.csv', parse_dates = ['date'], index_col = ['date'])
index_df = pd.read_csv('./data/index.csv', parse_dates = ['date'], index_col = ['date'])

fund_df = fund_df[fund_df.index >= '2012-01-01']
index_df = index_df[index_df.index >= '2012-01-01']

index_df = index_df.loc[fund_df.index]

fund_df = fund_df.fillna(method = 'pad')
index_df = index_df.fillna(method = 'pad')


columns = []
for code in fund_df.columns:
    columns.append(str(code) + '.OF')

fund_df.columns = columns

df = pd.concat([fund_df, index_df], axis = 1, join_axes = [fund_df.index])
df = df.fillna(-1)
df.to_csv('nav.csv')
