#coding=utf-8
'''
Created at Dec. 28, 2018
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import sys
import logging
sys.path.append('shell')
import numpy as np
import pandas as pd
from ipdb import set_trace
from trade_date import ATradeDate


logger = logging.getLogger(__name__)


def date_as_timestamp(date):

    if date is None:
        return None

    if isinstance(date, (int, np.int64)) and 18000101 <= date and date <= 29991231:
        date = str(date)

    return pd.Timestamp(date)

def date_as_caihui_str(date):

    if date is None:
        return None

    return date_as_timestamp(date).strftime('%Y%m%d')


def last_year(date):

    days_in_month = pd.Timestamp(date.year-1, date.month, 1).days_in_month
    return pd.Timestamp(date.year-1, date.month, min(date.day, days_in_month))

def last_quarter(date):

    if date.month < 4:
        days_in_month = pd.pd.Timestamp(date.year-1, date.month+9, 1).days_in_month
        return pd.Timestamp(date.year-1, date.month+9, min(date.day, days_in_month))
    else:
        days_in_month = pd.Timestamp(date.year, date.month-3, 1).days_in_month
        return pd.Timestamp(date.year, date.month-3, min(date.day, days_in_month))

def next_month(date):

    if date.month < 12:
        days_in_month = pd.Timestamp(date.year, date.month+1, 1).days_in_month
        return pd.Timestamp(date.year, date.month+1, min(date.day, days_in_month))
    else:
        days_in_month = pd.Timestamp(date.year+1, 1, 1).days_in_month
        return pd.Timestamp(date.year+1, 1, min(date.day, days_in_month))

def month_start(date):

    return pd.Timestamp(date.year, date.month, 1)

def end_of_last_half_year(date):

    return 0

def last_end_date_fund_skdetail_all_published(date):

    if date.month < 4:
        return pd.Timestamp(date.year-1, 6, 30)
    elif date.month < 9:
        return pd.Timestamp(date.year-1, 12, 31)
    else:
        return pd.Timestamp(date.year, 6, 30)

def last_end_date_fund_skdetail_ten_published(date):

    if date.month < 2:
        return pd.Timestamp(date.year-1, 9, 30)
    elif date.month < 5:
        return pd.Timestamp(date.year-1, 12, 31)
    elif date.month < 8:
        return pd.Timestamp(date.year, 3, 31)
    elif date.month < 11:
        return pd.Timestamp(date.year, 6, 30)
    else:
        return pd.Timestamp(date.year-1, 9, 30)

def trade_date_before(date):

    trade_dates = ATradeDate.trade_date()

    if date <= trade_dates[0]:
        raise ValueError('There is not trading day before the day.')

    lo = 0
    hi = trade_dates.shape[0]
    while lo + 1 < hi:
        mi = (lo + hi) // 2
        if date < trade_dates[mi]:
            hi = mi
        else:
            lo = mi

    if date == trade_dates[lo]:
        return trade_dates[lo-1]
    else:
        return trade_dates[lo]

def trade_date_not_later_than(date):

    trade_dates = ATradeDate.trade_date()

    if date < trade_dates[0]:
        raise ValueError('It is before the first day of trading.')

    lo = 0
    hi = trade_dates.shape[0]
    while lo + 1 < hi:
        mi = (lo + hi) // 2
        if date < trade_dates[mi]:
            hi = mi
        else:
            lo = mi

    return trade_dates[lo]
