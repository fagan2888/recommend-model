#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import pandas as pd
import numpy as np
from user_sta import max_dd
from db import base_ra_fund_nav, base_ra_fund
from ipdb import set_trace

pf_ret = 0.0718
pf_maxdd = -0.0462

#fn = base_ra_fund_nav.load_daily(begin_date = '2017-01-01', end_date = '2017-12-15')
fn = pd.read_csv('data/fund_data/fund_nav.csv', index_col = ['date'], parse_dates = True)
fn = fn.dropna(axis = 1)
ret = fn.apply(lambda x:x[-1]/x[0]-1, reduce = 1)
maxdd = fn.apply(max_dd, reduce = 1)
maxdd_lose= maxdd[maxdd > pf_maxdd]

ret_win = ret[ret < pf_ret]
ret_lose = ret[ret >= pf_ret]
#ret_lose = pd.DataFrame(ret_lose, index = ret_lose.index, columns = ['ret'])
vol = base_ra_fund.load()

vol = vol.loc[:, ['ra_code', 'ra_volume']]
vol = vol.set_index(['ra_code'])
vol = vol[vol.ra_volume > 2e8]
set_trace()

rl = vol.index.intersection(ret_lose.index)
al = rl.intersection(maxdd_lose.index)
set_trace()

print ret_lose
