#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import numpy as np
from db import base_ra_index_nav
from datetime import datetime, timedelta
from ipdb import set_trace


def load_ts_order():
    to = pd.read_csv('../data/user_data/ts_order.csv')
    return to

def load_sz_nav():
    sz = base_ra_index_nav.load_series('120000016')
    return sz

def load_ts_holding_nav():
    tn = pd.read_csv('../data/user_data/ts_holding_nav (1).csv')
    return tn

def cal_redem_date():
    to = load_ts_order()
    sz = load_sz_nav()
    tn  = load_ts_holding_nav()
    rd = to[to['ts_trade_type'] == 4][to['ts_trade_status'] == 6][to['ts_trade_date'] <= '2017-12-12']
    rd = rd[rd.ts_trade_date != '0000-00-00']
    rd = rd.sort_values(['ts_trade_date'])
    #rd = rd.ts_trade_date.values
    #redem_dates.sort()
    #redem_dates = redem_dates[110:]
    total = 0.0
    up = 0.0
    inc_range = []
    redem_dates = rd.ts_trade_date.values
    uids = rd.ts_uid.values
    for date, uid in zip(redem_dates, uids):
        tmp_nav = tn[tn.ts_date == date][tn.ts_uid == uid].ts_nav.values
        tmp_nav = tmp_nav[0] if len(tmp_nav > 0) else 0
        if tmp_nav > 1:
            inc_range.append(tmp_nav)
            total += 1.0
            day = datetime.strptime(date, '%Y-%m-%d').date()
            sz_before = sz[day]
            sz_after = sz[day + timedelta(30)]
            if sz_after > sz_before:
                up += 1.0
            ratio = up/total
            avg_inc = np.mean(inc_range)
            least_inc = np.min(inc_range)
            print total, up, ratio, tmp_nav, avg_inc, least_inc
        else:
            continue

    print ratio, avg_inc, least_inc

    print rd


def cal_range():
    to = load_ts_order()
    sz = load_sz_nav()
    tn  = load_ts_holding_nav()
    rd = to[to['ts_trade_type'] == 4][to['ts_trade_status'] == 6][to['ts_trade_date'] <= '2018-1-12']
    rd = rd[rd.ts_trade_date != '0000-00-00']
    rd = rd.sort_values(['ts_trade_date'])

    pd = to[to['ts_trade_type'] == 3][to['ts_trade_status'] == 6][to['ts_trade_date'] <= '2018-1-12']
    pd = pd[pd.ts_trade_date != '0000-00-00']
    pd = pd.sort_values(['ts_trade_date'])

    redem_dates = rd.ts_trade_date.values
    uids = rd.ts_uid.values
    total = 0.0
    success = 0.0
    for date, uid in zip(redem_dates, uids):
        rnav = tn[tn.ts_date == date][tn.ts_uid == uid].ts_nav.values
        rnav = rnav[0] if len(rnav > 0) else 0
        if rnav > 1:
            rsz = sz[date]
            pdate = pd[pd.ts_uid == uid]
            pdate = pdate[pdate.ts_trade_date >= date]
            if len(pdate) > 0:
                total += 1
                psz = sz[pdate.ts_trade_date.values[0]] 
                chg = psz/rsz - 1
                if chg < -0.007:
                    success += 1

            ratio = success/total
            print total, success, ratio, chg







if __name__ == '__main__':
    #load_ts_order()
    #cal_redem_date()
    cal_range()

