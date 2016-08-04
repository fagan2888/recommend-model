#coding=utf8

import string
import MySQLdb
from datetime import datetime
import pandas as pd
import numpy as np
import sys
sys.path.append('shell')
import Const
import DBData
import FundIndicator


interval = 26
dates = DBData.trade_dates('2010-01-01', '2016-06-30')

pvs = [1]

portfolio_vs    = []
portfolio_dates = []
i = 1

while True:

    start_date = dates[i]
    end_date = dates[i]
    if i + interval >= len(dates):
        end_date   = dates[len(dates) - 1]
    else:    
        end_date   = dates[i + interval]

    df = DBData.stock_fund_value(start_date, end_date)
    dfr = df.pct_change().fillna(0.0)
    codes = dfr.columns

    
    ds = dfr.index
    for d in ds:
        r = 0
        for code in codes:
            r = r + dfr.loc[d, code] * (1.0 / len(codes))
        pvs.append(pvs[-1] * (1 + r))
        portfolio_vs.append(pvs[-1])
        portfolio_dates.append(d)
        print d, pvs[-1]
    
    if i + interval >= len(dates):
        break

    i = i + interval

df  = pd.DataFrame(portfolio_vs, index = portfolio_dates, columns = ['nav'])
print df
df.to_csv('five_year_fund.csv')
print "sharpe : ", FundIndicator.portfolio_sharpe(df['nav'].values)
print "annual_return : ", FundIndicator.portfolio_return(df['nav'].values)
print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(df['nav'].values)


#df = DBData.stock_fund_value('2010-01-01', '2016-06-30')
#print df
