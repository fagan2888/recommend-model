#coding=utf8


import sys
sys.path.append('shell')
import pandas as pd
import DFUtil
import DBData
import numpy as np


if __name__ == '__main__':


    start_date = '2015-10-31'
    end_date   = '2016-10-31'

    df_nav_fund = DBData.stock_day_fund_value(start_date, end_date)
    df_inc_fund = df_nav_fund.pct_change().fillna(0.0)
    #print df_inc_fund
    df_nav_index = DBData.db_index_value_daily(start_date, end_date)
    df_nav_index = df_nav_index[['000905.SH']]
    df_inc_index = df_nav_index.pct_change().fillna(0.0)
    #print df_inc_index

    dates = df_inc_fund.index

    vs = []
    for col in df_inc_fund.columns:
        rise = []
        decline = []
        for d in dates:
            fr = df_inc_fund.loc[d, col]
            ir = df_inc_index.loc[d, '000905.SH']
            if ir >= 0.0:
                rise.append(fr - ir)
            else:
                decline.append(ir - fr)
        vs.append([np.mean(rise), np.mean(decline)])
        print col, 'done'

    df = pd.DataFrame(vs, index = df_inc_fund.columns, columns = ['rise', 'decline'])
    print df
    df.to_csv('fund_rise_decline.csv')
