#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import numpy as np
from db import base_ra_index_nav, asset_on_online_nav, asset_on_online_fund
from datetime import datetime, timedelta
from ipdb import set_trace

def cal_date():
    pf = asset_on_online_nav.load_series('800000', 8, begin_date = '2017-01-01')
    pf = pd.DataFrame(pf, index = pf.index)
    pf.columns = ['nav']
    pf['nav_diff'] = pf.nav.diff()
    pf = pf[pf.nav_diff != 0]
    del pf['nav_diff']

    pf['dd'] = pf.rolling(window = 500, min_periods = 1).apply(lambda x:x[-1]/max(x) - 1)
    dd1 = [0]
    dd2 = [0]
    dd3 = [0]
    dd4 = [0]
    for i in range(1, len(pf)):
        if (pf.dd[i] < -0.01) and (pf.dd[i-1] > -0.01):
            dd1.append(1)
        else:
            dd1.append(0)
    pf['dd1'] = dd1

    for i in range(1, len(pf)):
        if (pf.dd[i] < -0.02) and (pf.dd[i-1] > -0.02):
            dd2.append(1)
        else:
            dd2.append(0)
    pf['dd2'] = dd2

    for i in range(1, len(pf)):
        if (pf.dd[i] < -0.03) and (pf.dd[i-1] > -0.03):
            dd3.append(1)
        else:
            dd3.append(0)
    pf['dd3'] = dd3

    for i in range(1, len(pf)):
        if (pf.dd[i] < -0.04) and (pf.dd[i-1] > -0.04):
            dd4.append(1)
        else:
            dd4.append(0)
    pf['dd4'] = dd4

    pf['nav1'] = pf['nav'].shift(-1)
    pf['nav2'] = pf['nav'].shift(-2)
    pf['nav3'] = pf['nav'].shift(-3)
    pf['nav4'] = pf['nav'].shift(-4)
    pf['nav5'] = pf['nav'].shift(-5)
    pf['nav6'] = pf['nav'].shift(-6)
    pf['nav7'] = pf['nav'].shift(-7)
    pf['nav8'] = pf['nav'].shift(-8)
    pf['nav9'] = pf['nav'].shift(-9)
    pf['nav10'] = pf['nav'].shift(-10)

    for dd in ['dd1','dd2','dd3','dd4','dd5']:
        tmp_pf = pf[pf[dd] == 1]
        for nav in ['nav1', 'nav2', 'nav3', 'nav4', 'nav5', 'nav6', 'nav7', 'nav8', 'nav9', 'nav10']:
            tmp_ret = (tmp_pf[nav]/tmp_pf['nav'] - 1).mean()
            print dd, nav, tmp_ret

    return pf


def cal_ret():
    pf = asset_on_online_nav.load_series('800000', 8, begin_date = '2016-12-31')
    ret = pf.resample('m').last().pct_change()
    ret = ret.dropna()
    ret = 1+ret
    nav = 0
    for i in range(12):
        nav += ret[i-1:].prod()
    print nav
    print pf[-1]/pf[0]*12


def cal_ap():
    fp = asset_on_online_fund.load_fund_pos_code('800000')
    fp = fp.reset_index(level = 1)
    fp = fp[fp.index >= '2017']
    fn = pd.read_csv('data/fund_data/fund_nav.csv', index_col = ['date'], parse_dates = True)
    ap_date = fp.index.unique()
    ap_date = ap_date.drop(datetime(2017,2,24))
    ap_date = [
        datetime(2017,1,12), 
        datetime(2017,2,28), 
        datetime(2017,4,10), 
        datetime(2017,4,27), 
        datetime(2017,6,26), 
        datetime(2017,9,9), 
        datetime(2017,11,10), 
        datetime(2017,11,10), 
    ]
    nav_date = [
        datetime(2017,1,12), 
        datetime(2017,2,28), 
        datetime(2017,4,10), 
        datetime(2017,4,27), 
        datetime(2017,6,26), 
        datetime(2017,9,8), 
        datetime(2017,11,10), 
        datetime(2017,12,15), 
    ]
    print ap_date

    for i in range(len(ap_date) - 2):
        date1 = ap_date[i]
        date2 = ap_date[i + 1]
        date3 = ap_date[i + 2]

        date11 = nav_date[i]
        date21 = nav_date[i + 1]
        date31 = nav_date[i + 2]

        pre_fund = fp.loc[date1].on_fund_code
        aft_fund = fp.loc[date2].on_fund_code

        pre_pos = fp.loc[date1].on_fund_ratio
        aft_pos = fp.loc[date2].on_fund_ratio

        pre_nav_start = fn.loc[date21, pre_fund]
        pre_nav_end = fn.loc[date31, pre_fund]

        aft_nav_start = fn.loc[date21, aft_fund]
        aft_nav_end = fn.loc[date31, aft_fund]

        pre_nav = np.sum(pre_pos.values*(pre_nav_end / pre_nav_start).values)
        aft_nav = np.sum(aft_pos.values*(aft_nav_end / aft_nav_start).values)
        print date2, pre_nav, aft_nav


def contrast():
    fp = asset_on_online_fund.load_fund_pos_code('800000')
    fp = fp.reset_index(level = 1)
    fp = fp[fp.index >= '2017']
    fn = pd.read_csv('data/fund_data/fund_nav.csv', index_col = ['date'], parse_dates = True)
    ap_date = [
        datetime(2017,4,10), 
        datetime(2017,4,27), 
        datetime(2017,6,26), 
    ]
    nav_date = [
        datetime(2017,4,10), 
        datetime(2017,4,27), 
        datetime(2017,6,26), 
    ]
    print ap_date

    for i in range(len(ap_date) - 2):
        date1 = ap_date[i]
        date2 = ap_date[i + 1]
        date3 = ap_date[i + 2]

        date11 = nav_date[i]
        date21 = nav_date[i + 1]
        date31 = nav_date[i + 2]

        pre_fund = fp.loc[date1].on_fund_code
        aft_fund = fp.loc[date2].on_fund_code

        pre_pos = fp.loc[date1].on_fund_ratio
        aft_pos = fp.loc[date2].on_fund_ratio

        pre_nav_start = fn.loc[date21, pre_fund]
        pre_nav_end = fn.loc[date31, pre_fund]

        aft_nav_start = fn.loc[date21, aft_fund]
        aft_nav_end = fn.loc[date31, aft_fund]

        set_trace()
        pre_nav = np.sum(pre_pos.values*(pre_nav_end / pre_nav_start).values)
        aft_nav = np.sum(aft_pos.values*(aft_nav_end / aft_nav_start).values)
        print date2, pre_nav, aft_nav


def lag_ap():
    fp = pd.read_csv('on_online_fund.csv', parse_dates = ['on_date'])
    fp.on_date += timedelta(7)
    fp.to_csv("on_online_fund_shift7.csv", index = False)
    print fp.head()


if __name__ == '__main__':
#    cal_date()
#    cal_ret()
#    contrast()
    lag_ap()
