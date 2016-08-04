#coding=utf8


import sys
sys.path.append('shell')
import pandas as pd
import LabelAsset
import EqualRiskAssetRatio
import EqualRiskAsset
import HighLowRiskAsset
import os
import DB
import combine


if __name__ == '__main__':


    #start_date = '2010-01-05'
    #end_date = '2016-04-22'
    #LabelAsset.labelasset(start_date, end_date)


    #df = pd.read_csv('./data/gaopeng.csv', index_col = 'date', parse_dates = 'date')
    #df = pd.read_csv('./data/kunge.csv', index_col = 'date', parse_dates = 'date')

    #sep = 4
    #length = 6
    #df = pd.read_csv('./data/funds.csv', index_col = 'date', parse_dates = 'date')

    stock_num = 4

    df = pd.read_csv('./data/funds.csv', index_col = 'date', parse_dates = 'date')

    dfr = df.pct_change().fillna(0.0)
    week_df  = df.resample('W-FRI').last()
    week_dfr = week_df.pct_change().fillna(0.0)

    print week_dfr

    #print week_df


    EqualRiskAssetRatio.equalriskassetratio(week_dfr)
    EqualRiskAsset.equalriskasset(dfr)


    #HighLowRiskAsset.highlowriskasset(sep, length)
    #combine.combine()


