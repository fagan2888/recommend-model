#coding=utf8


import sys
sys.path.append('shell')
import string
import pandas as pd
import numpy as np
from datetime import datetime
import Portfolio as PF



df  = pd.read_csv('./data/fund.csv', index_col = 'date', parse_dates = 'date' )
#df  = df.resample('W-FRI').last().fillna(method = 'pad')
dfr = df.pct_change().fillna(0.0)
print df

cols = df.columns
highdfr = dfr[cols[0:3]]
#lowdfr  = dfr[cols[6:10]]


his_week = 13
interval = 5


dates = dfr.index
rs = []
ds = []
ws = []
position_ds   = []
position_data = []

for i in range(his_week + 1, len(dates)):

    d = dates[i]
    if (i - his_week - 1) % interval == 0:

        tmp_dfr = highdfr.iloc[i - his_week:i,:]
        #print tmp_dfr
        #uplimit   = [1.0, 1.0, 1.0, 1.0, 0.2, 1.0]
        #downlimit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        #risk, returns, highws, sharpe = PF.markowitz_r(highdfr, [downlimit, uplimit])
        risk, returns, highws, sharpe = PF.markowitz_r(tmp_dfr, None)
        ws = highws
        position_ds.append(d)
        position_data.append(ws)


    cols = highdfr.columns
    r = 0.0
    for j in range(0, len(cols)):
        r = r + highdfr.loc[d, cols[j]] * ws[j]

    rs.append(r)
    ds.append(d)

        #print highws

df = pd.DataFrame(rs, index = ds, columns = ['nav'])
#print df
df.to_csv('./wenjing.csv')

position_df = pd.DataFrame(position_data, index = position_ds, columns = highdfr.columns)
position_df.to_csv('./wenjing_position.csv')


'''
highlen = len(highdfr)
lowlen  = len(lowdfr)


highdfr = highdfr.iloc[highlen - 13: highlen,:]
lowdfr  = lowdfr.iloc[lowlen - 13: lowlen,:]


uplimit   = [1.0, 1.0, 1.0, 1.0, 0.2, 1.0]
downlimit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

risk, returns, highws, sharpe = PF.markowitz_r(highdfr, [downlimit, uplimit])
uplimit   = [1.0, 1.0, 1.0, 1.0]
downlimit = [0.0, 0.0, 0.0, 0.0]
risk, returns, lowws, sharpe = PF.markowitz_r(lowdfr, [downlimit, uplimit])
'''


