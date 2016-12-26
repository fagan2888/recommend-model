#coding=utf8


import sys
sys.path.append('shell')
import LabelAsset
import pandas as pd
import DFUtil
import DBData
import numpy as np


if __name__ == '__main__':


    #start_date = '2005-01-01'
    #end_date   = '2016-12-13'

    #indexdf = DBData.db_index_value_daily(start_date, end_date)
    #indexdf.to_csv('zz500.csv')
    zz500df = pd.read_csv('./data/risk_asset_allocation_nav.csv', index_col = 'date', parse_dates = ['date'])
    #zz500df = indexdf.loc[zz500df.index][['000905.SH']]
    zz500df.columns = ['nav']
    #zz500df = zz500df[['nav']]
    zz500dfr = zz500df.pct_change()
    #zz500df = zz500df / zz500df.iloc[0]
    zz500df_m = zz500df.resample('M', how='last')
    dates = zz500df_m.index

    print 'date, drawdown'
    period = 36
    for i in range(0, len(dates) - period):
        total_share = 0
        total_money = 0
        startd = dates[i]
        lastd = dates[i + period]
        tmp_dfr = zz500dfr[zz500dfr.index >= startd]
        tmp_dfr = tmp_dfr[tmp_dfr.index <= lastd]
        tmp_df = (tmp_dfr + 1).cumprod()
        maxdrawdown = (tmp_df / tmp_df.cummax() - 1).min()
        #tmp_df   = tmp_dfr.cumprod()
        #print tmp_df
        #fixinvest_r = total_share * zz500df_m.loc[lastd] / period - 1
        #r           = zz500df_m.loc[lastd] / zz500df_m.loc[startd] - 1
        print '%s,%f' % (startd, maxdrawdown)
