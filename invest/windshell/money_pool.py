#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import const
import Financial as fin
import stockfilter as sf
import stocktag as st
import portfolio as pf
import fundindicator as fi
import fund_selector as fs
import data
import datetime
from numpy import *
import fund_evaluation as fe
import pandas as pd


rf = const.rf



if __name__ == '__main__':

    start_date = '2007-01-05'
    end_date = '2016-04-22'

    moneydf = data.bonds()
    moneydfr = moneydf.pct_change().fillna(0.0)

    dates = moneydf.index

    tag = {}

    for i in range(156, len(dates)):

        if (i - 156) % 13 == 0:

            start_date = dates[i - 52].strftime('%Y-%m-%d')
            allocation_start_date = dates[i - 13].strftime('%Y-%m-%d')
            end_date = dates[i].strftime('%Y-%m-%d')

            allocation_funddf = data.bond_value(allocation_start_date, end_date)
            fund_codes, tag = fs.select_money(allocation_funddf)

            #print tag
        #print tag
        # print fund_codes

        d = dates[i]
        print d.strftime('%Y-%m-%d'), ',', moneydfr.loc[d, tag['sharpe1']], ',', moneydfr.loc[d, tag['sharpe2']]


# print tag
# allocation_funddf      = allocation_funddf[fund_codes]

