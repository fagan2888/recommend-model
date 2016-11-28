#coding=utf8


import sys
import pandas as pd
import numpy  as np
from datetime import datetime
import statsmodels.api as sm
import sklearn
from sklearn import preprocessing
from sklearn import linear_model


if __name__ == '__main__':


    manager_fund_performance_df = pd.read_csv('manager_fund_performance.csv', index_col = 'SECODE')
    fund_nav_df = pd.read_csv('./data/fund_value.csv', index_col = 'date')
    fund_nav_df.fillna(method = 'pad', inplace = True)
    #fund_nav_dfr = fund_nav_df.pct_change().fillna(0.0)
    #fund_nav_df.dropna(axis = 1, inplace = True)

    cols = fund_nav_df.columns
    codes = []
    for col in cols:
        codes.append((int)(col))

    mask = manager_fund_performance_df['FSYMBOL'].map( lambda x : x in codes)
    manager_fund_performance_df = manager_fund_performance_df.loc[mask]

    cols  = manager_fund_performance_df['FSYMBOL'].values
    codes = []


    for col in cols:
        codes.append('%06d' % col)


    fund_nav_df = fund_nav_df[codes]
    print fund_nav_df
    fund_nav_df.to_csv('fund_nav.csv')


    #print manager_fund_performance_df
    #print df

    #print mask
