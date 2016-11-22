#coding=utf8


import sys
sys.path.append('shell')
import pandas as pd
import DFUtil
import DBData
import numpy as np


if __name__ == '__main__':

    start_date = '2014-10-31'
    end_date   = '2016-10-31'

    df_nav_fund = DBData.stock_day_fund_value(start_date, end_date)
    df_inc_fund = df_nav_fund.pct_change().fillna(0.0)

    df_zscore   = []

    dates = df_inc_fund.index
    for d in dates:
        rs = df_inc_fund.loc[d]
        mean = np.mean(rs)
        std = np.std(rs)
        zscore_rs = []
        if std == 0:
            for r in rs:
                zscore_rs.append(r)
        else:
            for r in rs:
                zscore_rs.append((r - mean) / std)
        df_zscore.append(zscore_rs)

    df_zscore = pd.DataFrame(df_zscore, index = dates, columns = df_inc_fund.columns)
    #print df_zscore

    df_corr = df_zscore.corr()
    df_corr.to_csv('fund_corr.csv')
    print df_corr
