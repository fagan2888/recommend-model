#coding=utf8



import string
import pandas as pd
import numpy as np


if __name__ == '__main__':

    fund_df = pd.read_csv('./data/stock.csv', index_col='date', parse_dates=['date'])


    lines = open('./data/multi_factor_pool.csv','r').readlines()
    codes = []
    for line in lines:
        code = '%06d' % string.atoi(line.strip())
        codes.append(code)

    cs = []
    for code in codes:
        if code in set(fund_df.columns.values):
            cs.append(code)

    fund_df = fund_df[cs]
    #fund_df = fund_df[-52:]
    dfr = fund_df.pct_change().dropna()
    
    corr = dfr.corr()
    corr = np.mean(corr, axis = 1)
    print corr
    #dfr.to_csv('./tmp/multi_factor_fund.csv')
