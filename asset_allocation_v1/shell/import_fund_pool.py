#coding=utf8


import MySQLdb
import pandas as pd
import datetime



if __name__ == '__main__':

    stock_fund_df = pd.read_csv('./tmp/stock_fund.csv', index_col = 'date', parse_dates = ['date'])


    print 'category,date,code'
    for date in stock_fund_df.index:
        for col in stock_fund_df.columns:
            v = stock_fund_df.loc[date, col]
            if type(v) is float:
                continue
            #print date , col, stock_fund_df.loc[date, col]
            v = eval(stock_fund_df.loc[date, col])
            for code in v:
                print '%s,%s,%s' % (col, datetime.datetime.strftime(date, '%Y-%m-%d'), code)
