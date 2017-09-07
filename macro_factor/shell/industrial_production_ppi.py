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
    index_df = pd.read_csv('./data/index.csv', index_col = ['date'], parse_dates = ['date'])
    dates = pd.date_range(index_df.index[0], index_df.index[-1])
    index_df = index_df.reindex(dates)
    index_df = index_df.fillna(method = 'pad')
    index_df = index_df.reindex(df.index)
    #df['NH0100.NHF'] = index_df['NH0100.NHF']
    #df['NH0100.NHF_yoy'] = df['NH0100.NHF'].rolling(window = 13).apply(lambda x : x[-1] / x[0] - 1) * 100
    #df['NH0100.NHF_yoy_lagging1'] = df['NH0100.NHF_yoy'].shift(1)
    #df['NH0100.NHF_inc'] = df['NH0100.NHF'].pct_change()
    df['industrial_production_diff'] = df['industrial_production'].diff()
    df['ppi_diff'] = df['ppi'].diff()
    df['industrial_production_diff_lagging1'] = df['industrial_production_diff'].shift(1)
    df['industrial_production_diff_lagging2'] = df['industrial_production_diff'].shift(2)
    df['industrial_production_diff_lagging3'] = df['industrial_production_diff'].shift(3)
    df['industrial_production_diff_lagging4'] = df['industrial_production_diff'].shift(4)
    df['industrial_production_diff_lagging5'] = df['industrial_production_diff'].shift(5)
    df['industrial_production_diff_lagging6'] = df['industrial_production_diff'].shift(6)
    #df = df.dropna()
    #print index_df
    #print df
    #df['industrial_production_rank12'] = df['industrial_production'].rolling(window = 12).apply(lambda x : Series(x).rank().iloc[-1])
    #df['ppi_rank12'] = df['ppi'].rolling(window = 12).apply(lambda x : Series(x).rank().iloc[-1])
    #df['industrial_production_rank12_gaussian_filter_3'] = df['industrial_production_rank12'].rolling(window = 12).apply(lambda x : gaussian_filter(x, 3.0)[-1])
    #df['industrial_production_rank12_gaussian_filter_3_lagging6'] = df['industrial_production_rank12_gaussian_filter_3'].shift(6)
    #df['ppi_rank12_gaussian_filter_3'] = df['ppi_rank12'].rolling(window = 12).apply(lambda x : gaussian_filter(x, 3.0)[-1])
    #df['industrial_production_ma6'] = df['industrial_production'].rolling(window = 6).mean()
    #df['industrial_production_ma6_lagging3'] = df['industrial_production_ma6'].shift(3)
    #df['industrial_production_lagging6'] = df['industrial_production'].shift(6)
    #df['industrial_production_ma12'] = df['industrial_production'].rolling(window = 12).mean()
    #df['industrial_production_ma12_lagging6'] = df['industrial_production_ma12'].shift(6)
    #df['industrial_production_gaussian_filter_1'] = gaussian_filter(df['industrial_production'], 1.0)
    #df['industrial_production_gaussian_filter_2'] = gaussian_filter(df['industrial_production'], 2.0)
    #df['industrial_production_gaussian_filter_3'] = gaussian_filter(df['industrial_production'], 3.0)
    #df['industrial_production_gaussian_filter_1'] = df['industrial_production'].rolling(window = 12).apply(lambda x : gaussian_filter(x, 1.0)[-1])
    #df['industrial_production_gaussian_filter_2'] = df['industrial_production'].rolling(window = 12).apply(lambda x : gaussian_filter(x, 2.0)[-1])
    #df['industrial_production_gaussian_filter_3'] = df['industrial_production'].rolling(window = 12).apply(lambda x : gaussian_filter(x, 3.0)[-1])
    #df['industrial_production_gaussian_filter_3_lagging3'] = df['industrial_production'].rolling(window = 12).apply(lambda x : gaussian_filter(x, 3.0)[-1]).shift(3)
    #df['industrial_production_gaussian_filter_3_lagging6'] = df['industrial_production_gaussian_filter_3'].shift(6)
    #df['industrial_production_gaussian_filter_3_lagging-6'] = df['industrial_production'].rolling(window = 12).apply(lambda x : gaussian_filter(x, 3.0)[-1]).shift(-6)
    df = df.dropna()
    df.to_csv('tmp.csv')
    #print df
    #print df.corr()
    #dfr = df.pct_change().fillna(0.0)
    #print dfr.corr()
    #ser = df['industrial_production_gaussian_filter_3_lagging3']

    X = df['industrial_production_diff_lagging1'].values
    y = df['ppi_diff'].values
    X = sm.add_constant(X)

    model = sm.OLS(y, X)
    result = model.fit()
    print result.summary()


    '''
    for i in range(0 , 12):
        df['industrial_production_lagging'  + str(i)] = df['industrial_production'].shift(i)
        tmp_df = df.copy().dropna()
        #tmp_dfr = tmp_df.pct_change().dropna()
        mse = 0
        for n in range(0, len(tmp_df)):
            #print tmp_dfr.iloc[n]['industrial_production_lagging'  + str(i)]
            #print tmp_dfr.iloc[n]['ppi']
            mse = mse + (tmp_df.iloc[n]['industrial_production_lagging'  + str(i)] - tmp_df.iloc[n]['NH0100.NHF_yoy']) ** 2
        print i, (mse / len(tmp_df)) ** 0.5
    '''
