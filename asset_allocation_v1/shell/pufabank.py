#coding=utf8


import numpy  as np
import pandas as pd
import time
import sys
sys.path.append('shell')
import FundValue
import DBData
import Portfolio



'''
def allocation(dfr):
    final_risk, final_return, final_ws, final_sharp = Portfolio.markowitz_r(dfr, None)
    ws = {}
    for i in range(0, len(dfr.columns)):
        code = dfr.columns[i]
        #print code
        ws[code] = final_ws[i]
    return ws
'''



if __name__ == '__main__':


    '''
    start_date = '2014-01-01'
    end_date   = '2016-10-31'


    #codes = ['003152','003743','110001','163412','233005','320006','320013','519118','519127','630011']
    stock_codes = ['110001','163412','233005','320006','519118','519127','630011']
    gold_codes = ['320013']
    df = DBData.db_fund_value_daily(start_date, end_date, codes)
    #print df

    df = df.resample('W-FRI').last()
    df = df.fillna(method = 'pad')
    df = df / df.iloc[0]
    #print df
    dfr = df.pct_change().fillna(0.0)

    #print dfr
    his_back = 26
    interval = 13
    result_df = FundValue.FundValue(dfr, his_back, interval, allocation)

    result_df.to_csv('pufavdf.csv')
    '''


    print 'category,date,code,jensen,ppw,sharpe,sortino,stability'
    codes = ['003152','003743','110001','163412','233005','320006','320013','519118','519127','630011']
    stock_codes = ['110001','163412','320006','630011']
    bond_codes = ['233005','519127','519118']
    gold_codes = ['320013']
    start_date = '2010-01-08'
    end_date   = '2016-10-31'
    dates = DBData.trade_dates(start_date, end_date)
    his_back = 26
    interval = 13
    for i in range(his_back, len(dates)):
        if i % interval == 0:
            d = dates[i - his_back]
            for code in stock_codes:
                print '%s,%s,%s,0,0,0,0,0' % ('largecap', d, code)
            for code in stock_codes:
                print '%s,%s,%s,0,0,0,0,0' % ('smallcap', d, code)
            for code in stock_codes:
                print '%s,%s,%s,0,0,0,0,0' % ('rise', d, code)
            for code in stock_codes:
                print '%s,%s,%s,0,0,0,0,0' % ('oscillation', d, code)
            for code in stock_codes:
                print '%s,%s,%s,0,0,0,0,0' % ('decline', d, code)
            for code in stock_codes:
                print '%s,%s,%s,0,0,0,0,0' % ('growth', d, code)
            for code in stock_codes:
                print '%s,%s,%s,0,0,0,0,0' % ('value', d, code)
            for code in gold_codes:
                print '%s,%s,%s,0,0,0,0,0' % ('GLNC', d, code)
            print '%s,%s,%s,0,0,0,0,0' % ('money', d, '000509')
            for code in bond_codes:
                print '%s,%s,%s,0,0,0,0,0' % ('ratebond', d, code)
            for code in bond_codes:
                print '%s,%s,%s,0,0,0,0,0' % ('creditbond', d, code)
            #print '%s,%s,%s,0,0,0,0,0' % ('ratebond', d, '003152')
            #print '%s,%s,%s,0,0,0,0,0' % ('creditbond', d, '003152')
