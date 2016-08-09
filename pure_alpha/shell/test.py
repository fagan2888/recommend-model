#coding=utf8


import pandas as pd
import string


fund_df = pd.read_csv('./data/stock.csv', index_col='date', parse_dates=['date'])
fund_df = fund_df[-52:]
fund_dfr = fund_df.pct_change().fillna(0.0)


fund_pool_lines = open('./data/fund_pool.csv','r').readlines()
fund_pool = []
for line in fund_pool_lines:
    code = '%06d' % string.atoi(line.strip())
    if code in set(fund_dfr.columns.values):
        fund_pool.append(code)

measure_fund_pool = set(fund_pool)


fund_pool_lines = open('./data/after_results.csv','r').readlines()
fund_pool = []
n = 0
for line in fund_pool_lines:
    code = '%06d' % string.atoi(line.strip())
    if code in set(fund_dfr.columns.values):
        fund_pool.append(code)
    n = n + 1
    if n > 132:
        break

alpha_fund_pool = set(fund_pool)


_measure_fund_pool = measure_fund_pool.difference(alpha_fund_pool)
_alpha_fund_pool   = alpha_fund_pool.difference(measure_fund_pool)


l = len(_measure_fund_pool)
print l
dates = fund_dfr.index
rs    = []
for d in dates:
    r = 0.0
    for code in measure_fund_pool:
        r = r + fund_dfr.loc[d, code] / l
    rs.append(r)

df = pd.DataFrame(rs, index = dates, columns = ['r'])
df.to_csv('pool.csv')


l = len(_alpha_fund_pool)
print l
dates = fund_dfr.index
rs    = []
for d in dates:
    r = 0.0
    for code in alpha_fund_pool:
        r = r + fund_dfr.loc[d, code] / l
    rs.append(r)


df = pd.DataFrame(rs, index = dates, columns = ['r'])
df.to_csv('after_results.csv')
