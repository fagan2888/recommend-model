#coding=utf8


import sys
sys.path.append('shell')
import LabelAsset
import pandas as pd
import DFUtil
import DBData
import numpy as np


if __name__ == '__main__':


    start_date = '2014-06-30'
    end_date   = '2016-10-31'


    codes = ['210009', '257070','240016', '510230', '519983', '001882', '000309']
    stockdf_nav_fund = DBData.db_fund_value_daily(start_date, end_date, codes)
    #df_nav_fund = df_nav_fund / df_nav_fund.iloc[0]
    #print df_nav_fund
    #df_nav_fund.to_csv('stock.csv')

    codes = ['530008', '000206', '233005']
    bonddf_nav_fund = DBData.db_fund_value_daily(start_date, end_date, codes)
    #df_nav_fund = df_nav_fund / df_nav_fund.iloc[0]
    #print df_nav_fund
    #df_nav_fund.to_csv('bond.csv')

    codes = ['000300.SH']
    indexdf_nav_fund = DBData.db_index_value_daily(start_date, end_date, codes)
    #df_nav_fund = df_nav_fund / df_nav_fund.iloc[0]
    #print df_nav_fund
    #df_nav_fund.to_csv('index.csv')

    df = pd.concat([stockdf_nav_fund, bonddf_nav_fund, indexdf_nav_fund], axis = 1)
    print df
    df.to_csv('stock_bond_index.csv')
