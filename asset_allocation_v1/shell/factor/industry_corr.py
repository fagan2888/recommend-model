#coding=utf8



import pandas as pd
import sys
sys.path.append('shell')
import config
import DBData
import statsmodels.api as sm
import numpy as np
import time



if __name__ == '__main__':


    start_date = '2014-01-01'
    end_date   = '2016-10-30'


    industry_df = pd.read_csv('./data/industry_index.csv', index_col = ['date'], parse_dates = ['date'])
    #industry_df = pd.read_csv('./data/factor_index.csv', index_col = ['date'], parse_dates = ['date'])
    industry_dfr = industry_df.pct_change().fillna(0.0)
    #print industry_dfr

    indexcols = []
    for i in range(1, 29):
        indexcols.append(str(i) + '.ix')

    industry_dfr.columns = indexcols
    #df = DBData.stock_day_fund_value(start_date, end_date)
    #df.to_csv('./data/test_fund.csv')
    fund_df = pd.read_csv('./data/test_fund.csv', index_col = ['date'], parse_dates = ['date'])
    fund_dfr = fund_df.pct_change().fillna(0.0)
    dates = fund_dfr.index & industry_dfr.index
    back = 252
    industry_dfr = industry_dfr.loc[dates]
    fund_dfr = fund_dfr.loc[dates]

    #print industry_dfr.index
    #print dfr.index
    #allfund_results = []
    #allparams_results = []
    for col in fund_dfr.columns:

        dfr = fund_dfr[[col]]

        results = []
        params = []
        ds = []
        total = 0
        correct = 0
        bias = 0
        rsquared_adj_sum = 0
        n = 0
        corr = None

        for i in range(back, len(dates) - 1):
            d = dates[i]
            industry_r = industry_dfr.iloc[i - back : i]
            fund_r     = dfr.iloc[i - back : i]

            tmpdf = pd.concat([industry_r, fund_r], axis = 1)
            #print tmpdf
            corr_df = tmpdf.corr()
            corr_df = corr_df[col]
            if corr is None:
                corr = corr_df
            else:
                corr = corr + corr_df
            n = n + 1

        print corr / n

            #corr_df = corr_df[col]
            #print corr_df
