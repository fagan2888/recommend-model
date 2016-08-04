#coding=utf8


import sys
sys.path.append('shell')
import string
import pandas as pd
import numpy as np
from datetime import datetime
import Portfolio as PF



df  = pd.read_csv('./data/hh.csv', index_col = 'date', parse_dates = 'date' )
df  = df.resample('W-FRI').last()
dfr = df.pct_change().fillna(0.0)


cols = df.columns
highdfr = dfr[cols[0:6]]
lowdfr  = dfr[cols[6:10]]


his_week = 13
interval = 13


dates = dfr.index
for i in range(his_week + 1, len(dates)):

	if (i - his_week - 1) % interval == 0:

		tmp_dfr = highdfr.iloc[i - his_week:i,:]

		uplimit   = [1.0, 1.0, 1.0, 1.0, 0.2, 1.0]
		downlimit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

		risk, returns, highws, sharpe = PF.markowitz_r(highdfr, [downlimit, uplimit])
		
		print highws



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


