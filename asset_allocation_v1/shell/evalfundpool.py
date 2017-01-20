#coding=utf8


import pandas as pd
import sys
sys.path.append('shell')
import DBData
import datetime
import numpy as np


if __name__ == '__main__':

    tdates = DBData.all_trade_dates_daily()
    tdates = tdates[0:-10]

    fundpool = {}
    lines = open('tmp/factor_fundpool.csv','r').readlines()
    for line in lines:
        line = line.strip()
        vec = line.split()
        date = vec[0].strip()
        codes = []
        for i in range(1, len(vec)):
            codes.append(vec[i].strip())
        fundpool[date] = codes

    stock_fund_df = pd.read_csv('./tmp/stock_fund.csv', index_col = ['date'])
    stock_fund_df = stock_fund_df.reindex(tdates)
    stock_fund_df = stock_fund_df.fillna(method = 'pad')

    pre_fundpool = None
    dfr = None
    turnover = []
    corrs = []
    ds = []
    rs = []
    started = False
    dates = fundpool.keys()
    dates.sort()
    for i in range(0, len(tdates) - 1):
        d = tdates[i]
        count = dates.count(d)
        if count == 1:
            codes = fundpool[d]
            codes = set(codes)

            #print d, len(codes)
            #allcs = set()
            #for col in stock_fund_df.columns:
            #    cs = set(eval(stock_fund_df.loc[d, col]))
            #    allcs = allcs | cs
                #print d, col, len(cs)
            #print d, len(codes)
            #codes = codes & allcs
            codes = list(codes)

            start_date = tdates[i - 250]
            if i + 250 >= len(tdates):
                end_date = tdates[len(tdates) - 1]
            else:
                end_date = tdates[i + 250]


            if pre_fundpool is None:
                pre_fundpool = codes
            else:
                diff = set(pre_fundpool).difference(set(codes))
                turnover.append(1.0 * len(diff) / len(pre_fundpool))
                pre_fundpool = codes
                #print start_date, end_date, codes

            funddf = DBData.db_fund_value_daily(start_date, end_date, codes)
            funddfr = funddf.pct_change().fillna(0.0)
            dfr = funddfr
            back_funddfr = funddfr.loc[funddfr.index <= d]
            back_funddfr_corr_df = back_funddfr.corr()
            print d
            #print back_funddfr_corr_df
            corr = back_funddfr_corr_df.mean().mean()
            corrs.append(corr)
            #print d, corr

        if not (dfr is None):
            rs.append(dfr.loc[d].mean())
            ds.append(d)

    dfr = pd.DataFrame(rs, index = ds, columns = ['nav'])
    dfr.index.name = 'date'
    df = (dfr + 1).cumprod()
    print df

    df.to_csv('./tmp/factor_vs.csv')
    print np.mean(corrs), corrs
    print np.mean(turnover), turnover


    '''
    for d in dates:
        codes = fundpool[d]
        index = tdates.index(d)
        start_date =
    '''
