#coding=utf8


import sys
sys.path.append('shell')
import LabelAsset
import pandas as pd
import DFUtil
import DBData
import numpy as np


if __name__ == '__main__':


    stock_fund_level1_df = pd.read_csv('./tmp/stock_fund_level1.csv', index_col = 'date', parse_dates = ['date'])
    #stock_fund_level2_df = pd.read_csv('./tmp/stock_fund_level2.csv', index_col = 'date', parse_dates = ['date'])
    #print stock_fund_df

    largecap_fund = stock_fund_level1_df['largecap']
    #print largecap_fund
    dates = stock_fund_level1_df.index
    fund1 = set(eval(largecap_fund[-4]))
    fund2 = set(eval(largecap_fund[-3]))
    fund3 = set(eval(largecap_fund[-2]))
    fund4 = set(eval(largecap_fund[-1]))
    print fund4
    #print fund1 & fund2 & fund3 & fund4
    #for i in range(4, len(dates)):
    #    d = dates[i]
