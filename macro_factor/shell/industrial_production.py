#coding=utf8

import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import Series
from scipy.ndimage.filters import gaussian_filter
import statsmodels.api as sm


if __name__ == '__main__':


    df = pd.read_csv('./data/industrial_production_ppi.csv', index_col = ['date'], parse_dates = ['date'])
    for k,v in df.groupby(df.index.year):
        df['industrial_production'][(df.index.year == k) & ((df.index.month == 2) | (df.index.month == 1))] = v[(v.index.month == 1) | (v.index.month == 2)].mean()['industrial_production']

    #df = df[['industrial_production']]
    df = df.iloc[39 : ] / 100
    for k ,v in df.groupby(df.index.month):
        v = (v + 1).cumprod()
        df.loc[v.index] = v
    df_inc = df.pct_change()
    print df_inc
    df.to_csv('tmp.csv')
