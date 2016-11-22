#coding=utf8


import sys
sys.path.append('shell')
import LabelAsset
import pandas as pd
import DFUtil
import DBData
import numpy as np


if __name__ == '__main__':


    start_date = '2015-09-30'
    end_date   = '2016-09-30'

    stock_fund_df = pd.read_csv('./tmp/stock_fund.csv', index_col = 'date', parse_dates = ['date'])
    #print stock_fund_df
    codes = set()
    for date in stock_fund_df.index:
        for col in stock_fund_df.columns:
            #print type(stock_fund_df.loc[date, col])
            v = stock_fund_df.loc[date, col]
            if type(v) is float:
                continue
            #print date , col, stock_fund_df.loc[date, col]
            v = eval(stock_fund_df.loc[date, col])
            for code in v:
                codes.add('%06d' % (int)(code))

    #print len(codes)
    df_nav_fund = DBData.db_fund_value_daily(start_date, end_date, list(codes))
    df_inc_fund = df_nav_fund.pct_change().fillna(0.0)
    dates = df_inc_fund.index
    rs = []
    for i in range(0, len(dates)):
        r = np.mean(df_inc_fund.iloc[i].values)
        rs.append(r)
    sharpe = (np.mean(rs) * 52 - 0.025) / (np.std(rs) * (52 ** 0.5))
    print sharpe
    #print df_inc_fund
    #print np.mean(df_inc_fund)
    #corr_df = df_inc_fund.corr()
    #corr_df.to_csv('corr.csv')
    #print (np.sum(corr_df.iloc[0].values) - 1) / len(corr_df.columns)


    df_nav_fund = DBData.stock_day_fund_value(start_date, end_date)
    df_inc_fund = df_nav_fund.pct_change().fillna(0.0)
    dates = df_inc_fund.index
    rs = []
    for i in range(0, len(dates)):
        r = np.mean(df_inc_fund.iloc[i].values)
        rs.append(r)
    sharpe = (np.mean(rs) * 52 - 0.025) / (np.std(rs) * (52 ** 0.5))
    print sharpe
    #corr_df = df_inc_fund.corr()
    #print (np.sum(corr_df.iloc[0].values) - 1) / len(corr_df.columns)
