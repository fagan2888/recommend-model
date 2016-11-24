#coding=utf8


import sys
sys.path.append('shell')
import LabelAsset
import pandas as pd
import DFUtil
import DBData
import numpy as np


if __name__ == '__main__':


    stock_fund_level1_df = pd.read_csv('./tmp/stock_fund_level0.csv', index_col = 'date', parse_dates = ['date'])
    stock_fund_level2_df = pd.read_csv('./tmp/stock_fund_level2.csv', index_col = 'date', parse_dates = ['date'])
    #print stock_fund_df

    level1 = {}
    level2 = {}

    for date in stock_fund_level1_df.index:
        codes = set()
        for col in stock_fund_level1_df.columns:
            #print type(stock_fund_df.loc[date, col])
            v = stock_fund_level1_df.loc[date, col]
            if type(v) is float:
                continue
            #print date , col, stock_fund_df.loc[date, col]
            v = eval(stock_fund_level1_df.loc[date, col])
            for code in v:
                codes.add(code)
        level1[date] = codes


    for date in stock_fund_level2_df.index:
        codes = set()
        for col in stock_fund_level2_df.columns:
            #print type(stock_fund_df.loc[date, col])
            v = stock_fund_level2_df.loc[date, col]
            if type(v) is float:
                continue
            #print date , col, stock_fund_df.loc[date, col]
            v = eval(stock_fund_level2_df.loc[date, col])
            for code in v:
                codes.add(code)
        level2[date] = codes


    v = 0
    n = 0
    dates = stock_fund_level1_df.index
    for i in range(1 , len(dates)):

        d = dates[i]
        pre_d = dates[i - 1]

        level1pool = level1[d]
        level2pool = level2[d]
        level1prepool = level1[pre_d]
        level2prepool = level2[pre_d]

        #print len(level2pool.difference(level1pool))
        #diff = level2prepool.difference(level2pool)
        #diff = level1prepool.difference(level1pool)
        #diff = level1prepool.difference(level1pool)
        #diff = level2prepool.difference(level2pool)
        #print level2prepool
        #print level2pool
        #print diff
        #print
        #print d, 1.0 * len(level1pool) / len(level2pool)
        #print d, 1.0 * len(diff) / len(level2prepool)
        level1prel = len(level1prepool)
        tmp_level2pre = set(list(level2prepool)[0:level1prel])
        level1l = len(level1pool)
        tmp_level2 = set(list(level2pool)[0:level1l])

        diff = tmp_level2pre.difference(tmp_level2)

        #level2diff = diff.difference(level2pool)
        print d, 1.0 * len(diff) / len(tmp_level2pre)
        #print d, 1.0 * len(level2diff) / len(level1prepool)
        #print d, len(level1prepool)
        #print d, 1.0 * len(diff) / len(level1prepool)
        #v = v + 1.0 * len(diff) / len(level2prepool)
        v = v + 1.0 * len(diff) / len(tmp_level2pre)
        n = n + 1
        #print d, len(diff), len(level2prepool), len(level2pool), 1.0 * len(diff & level1prepool) / len(level1prepool)
        #print level1pool, level2pool, level1prepool, level2prepool
    print v / n
