#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import numpy as np
from db import base_ra_index_nav, asset_on_online_nav
from ipdb import set_trace

def load_benchmark_nav():
    sz = base_ra_index_nav.load_series('120000016', begin_date = '2017-01-01')
    pf = asset_on_online_nav.load_series('800000', 8, begin_date = '2017-01-01')

    sz = sz.reset_index()
    pf = pf.reset_index()

    sz['sz_ret'] = sz.nav.pct_change()
    pf['pf_ret'] = pf.on_nav.pct_change()

    sz = sz.fillna(0.0)
    pf = pf.fillna(0.0)
    #print pf
    return sz, pf

def load_user_nav():
    sz, pf = load_benchmark_nav()

    un = pd.read_csv('data/user_data/ts_holding_nav.csv', parse_dates = ['ts_date'])
    un = un[un.ts_date >= '2017-09-15']
    un = un.sort_values(['ts_date'])
    set_trace()

    un = pd.merge(un, sz, left_on = 'ts_date', right_on = 'date')
    un = pd.merge(un, pf, left_on = 'ts_date', right_on = 'on_date')
    un = un.loc[:, ['ts_uid', 'ts_nav', 'pf_ret', 'sz_ret']]
    #set_trace()
    un_maxdd_user = un.ts_nav.groupby(un.ts_uid).apply(max_dd)
    un_maxdd_sz = un.sz_ret.groupby(un.ts_uid).apply(max_dd_ret)
    un_maxdd_pf = un.pf_ret.groupby(un.ts_uid).apply(max_dd_ret)
    un_maxdd = pd.concat([un_maxdd_user, un_maxdd_sz, un_maxdd_pf], 1)
    un_maxdd.to_csv('data/user_data/user_maxdd.csv', index_label = 'uid')
    print un_maxdd

def max_dd(nav):
    nav = nav.values
    maxdd = 0
    for i in range(1, len(nav)):
        dd = nav[i]/max(nav[:i+1]) - 1
        if dd < maxdd:
            maxdd = dd

    return maxdd

def max_dd_ret(ret):
    ret = ret.values
    nav = np.cumprod(1+ret)
    #set_trace()
    maxdd = 0
    for i in range(1, len(nav)):
        dd = nav[i]/max(nav[:i+1]) - 1
        if dd < maxdd:
            maxdd = dd

    return maxdd


def maxdd_sta():
    maxdd = pd.read_csv('data/user_data/user_maxdd.csv', index_col = 0)
    print len(maxdd)
    print maxdd[maxdd]
#    print maxdd

if __name__ == '__main__':
#    load_benchmark_nav()
    load_user_nav()
#    maxdd_sta()
