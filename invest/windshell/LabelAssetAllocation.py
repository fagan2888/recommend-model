#coding=utf8


import string
import pandas as pd
import numpy as np
import fundindicator
from datetime import datetime
import portfolio as pf
import fundindicator as fi


dfr         = pd.read_csv('./tmp/labelasset.csv', index_col = 'date', parse_dates = 'date' )
dates = dfr.index

his_week = 13
interval = 1

result_dates = []
result_datas = []

position_dates = []
position_datas = []


portfolio_vs = [1]
result_dates.append(dates[his_week])

fund_values  = {}
fund_codes   = []


for i in range(his_week + 1, len(dates)):


    if (i - his_week - 1 ) % interval == 0:

        start_date = dates[ i- his_week].strftime('%Y-%m-%d')
        end_date   = dates[i - 1].strftime('%Y-%m-%d')

        allocation_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d')]
        allocation_dfr = allocation_dfr[allocation_dfr.index >= datetime.strptime(start_date, '%Y-%m-%d')]

        risk, returns, ws, sharpe = pf.markowitz_r(allocation_dfr, None)
        fund_codes = allocation_dfr.columns


        last_pv = portfolio_vs[-1]
        fund_values = {}
        for n in range(0, len(fund_codes)):
            fund_values[n] = [last_pv * ws[n]]

        position_dates.append(end_date)
        position_datas.append(ws)


    pv = 0
    d = dates[i]
    for n in range(0, len(fund_codes)):
        vs = fund_values[n]
        code = fund_codes[n]
        fund_last_v = vs[-1]
        fund_last_v = fund_last_v + fund_last_v * dfr.loc[d, code]
        vs.append(fund_last_v)
        pv = pv + vs[-1]
    portfolio_vs.append(pv)
    result_dates.append(d)

    print d , pv


result_datas  = portfolio_vs
result_df = pd.DataFrame(result_datas, index=result_dates,
                         columns=['all_asset'])

result_df.to_csv('./tmp/labelasset_net_value.csv')


position_df = pd.DataFrame(position_datas, index=position_dates, columns=dfr.columns)
position_df.to_csv('./tmp/labelasset_position.csv')

print "sharpe : ", fi.portfolio_sharpe(result_df['all_asset'].values)
print "annual_return : ", fi.portfolio_return(result_df['all_asset'].values)
print "maxdrawdown : ", fi.portfolio_maxdrawdown(result_df['all_asset'].values)
