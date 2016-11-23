#coding=utf8


import sys
sys.path.append('shell')
import LabelAsset
import pandas as pd
import DFUtil
import DBData
import numpy as np


if __name__ == '__main__':


    #start_date = '2014-06-30'
    start_date = '2012-06-30'
    end_date   = '2016-10-31'

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
                codes.add(code)

    codes = list(codes)
    dates = stock_fund_df.index
    ratios = []
    for d in dates:
        cs = stock_fund_df.loc[d]
        d_cs = set()
        for vs in cs.values:
            if type(vs) is float:
                continue
            for v in eval(vs):
                d_cs.add(v)
        ratio = []
        for code in codes:
            if code in d_cs:
                ratio.append(1.0 / len(d_cs))
            else:
                ratio.append(0)
        ratios.append(ratio)

    #print ratios
    cs = []
    for code in codes:
       cs.append('%06d' % (int)(code))
    #print cs
    df_position = pd.DataFrame(ratios, index = dates, columns = cs)
    df_position.index.name = 'date'
    df_position.to_csv('position.csv')
    '''
    df_position.columns.name = 'fund'
    df_position = df_position.unstack()
    #print df_position.head()
    df_position = df_position.to_frame('ratio')
    print 'aa' , df_position.head()
    df_position = df_position[ df_position['ratio'] > 0.0]
    df_position['risk'] = 1.0
    df_position['category'] = 11
    df_position = df_position.reset_index().set_index(['risk','date','category','fund'])
    df_position.sort_index(inplace = True)
    df_position.to_csv('position.csv')
    '''
    #print df_position.columns
    #print df_position
    #print df_position
    df_nav_fund = DBData.db_fund_value_daily(start_date, end_date, df_position.columns)
    df_inc_fund = df_nav_fund.pct_change().fillna(0.0)
    #print df_inc_fund
    df_nav_portfolio = DFUtil.portfolio_nav(df_inc_fund, df_position, result_col='portfolio')
    print df_nav_portfolio
    df_nav_portfolio['portfolio'].to_csv('fund_pool.csv')
    #print df_position


    pre_fund_pool = None
    dates = stock_fund_df.index
    for d in dates:
        cs = stock_fund_df.loc[d]
        d_cs = set()
        for vs in cs.values:
            if type(vs) is float:
                continue
            for v in eval(vs):
                d_cs.add(v)
        if pre_fund_pool is None:
            pre_fund_pool = d_cs
        else:
            ratio = 1.0 - 1.0 * len(pre_fund_pool & d_cs) / len(pre_fund_pool)
            print d, ratio
            pre_fund_pool = d_cs
    '''

    codes = set()
    dates = stock_fund_df.index.values
    #print dates
    ds = []
    columns = stock_fund_df.columns
    rs = {}
    for i in range(0 ,len(dates) - 1):
        start_date = dates[i]
        end_date = dates[i + 1]
        for col in stock_fund_df.columns:
            #print type(stock_fund_df.loc[date, col])
            v = stock_fund_df.loc[date, col]
            if type(v) is float:
                continue
            #print date , col, stock_fund_df.loc[date, col]
            v = eval(stock_fund_df.loc[date, col])
            cs = []
            for c in v:
                cs.append('%06d' % (int)(c))
            df_nav_fund = DBData.db_fund_value_daily(start_date, end_date, cs)
            df_inc_fund = df_nav_fund.pct_change()
            for d in df_inc_fund.index:
                fr = df_inc_fund.loc[d]
                ds.append(d)
                r = 0
                for item in fr:
                    r = r + 1.0 * item / len(fr)
                tmp = rs.setdefault(d, {})
                tmp[col] = r


    dates = rs.keys()
    dates = list(dates)
    dates.sort()
    fund_incs = []
    for d in dates:
        incs = []
        r = rs[d]
        for code in columns:
            if code in r.keys():
                incs.append(r[code])
            else:
                incs.append(0.0)
        fund_incs.append(incs)

    df = pd.DataFrame(fund_incs, index = dates, columns = columns)
    df.dropna(inplace = True)
    corr_df = df.corr()
    corr_df.to_csv('corr.csv')
    print corr_df
    #df.to_csv('label_asset.csv')
            #for code in v:
            #    codes.add(code)
    '''
