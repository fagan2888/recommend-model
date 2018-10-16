# -*- coding: utf-8 -*-

import string
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import os
import time
import logging
import re
import util_numpy as npu
import MySQLdb
import config
from ipdb import set_trace
import matplotlib.pyplot as plt


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func, literal_column
from tabulate import tabulate
from db import database, base_exchange_rate_index, base_ra_index, asset_ra_pool_fund, base_ra_fund, asset_ra_pool, asset_on_online_nav, asset_ra_portfolio_nav, asset_on_online_fund, asset_mz_markowitz_nav, asset_mz_markowitz_pos
from db import base_fund_infos, base_company_infos, asset_fund, base_ra_fund_nav
from util import xdict
from trade_date import ATradeDate
from asset import Asset
from monetary_fund_filter import MonetaryFundFilter

import traceback, code

logger = logging.getLogger(__name__)


def cal_month_wr(allocate_inc, df_inc):

    # fund_month_inc = df_inc.groupby(df_inc.index.strftime('%Y-%m')).sum()
    # allocate_month_inc = allocate_inc.groupby(allocate_inc.index.strftime('%Y-%m')).sum()

    fund_month_inc = df_inc.resample('m').sum()
    allocate_month_inc = allocate_inc.resample('m').sum()

    ranks = []
    for date in allocate_month_inc.index:
        allocate_r = allocate_month_inc.loc[date]
        fund_month_r = fund_month_inc.loc[date].ravel()
        fund_month_r = list(fund_month_r[fund_month_r > 0.0])
        fund_month_r.append(allocate_r)
        fund_month_r.sort(reverse=True)
        # print(fund_month_r.index(allocate_r), len(fund_month_r), 1.0 * fund_month_r.index(allocate_r) / len(fund_month_r))
        ranks.append(1.0 * fund_month_r.index(allocate_r) / len(fund_month_r))
    print(np.mean(ranks))
    df_month_wr = pd.Series(data=ranks, index=allocate_month_inc.index)
    df_month_wr = df_month_wr.to_frame('wr')
    df_month_wr = df_month_wr[df_month_wr.index > '2012-12-31']
    df_month_wr.to_csv('data/df_month_wr.csv', index_label='date')

    return df_month_wr


def cal_month_money_wr(allocate_inc, df_inc, fund_scale):

    # fund_month_inc = df_inc.groupby(df_inc.index.strftime('%Y-%m')).sum()
    # allocate_month_inc = allocate_inc.groupby(allocate_inc.index.strftime('%Y-%m')).sum()

    fund_month_inc = df_inc.resample('m').sum()
    allocate_month_inc = allocate_inc.resample('m').sum()

    ranks = []
    for date in allocate_month_inc.index:

        allocate_r = allocate_month_inc.loc[date]
        fund_month_r = fund_month_inc.loc[date]
        fund_month_r = fund_month_r.replace(0.0, np.nan).dropna()
        valid_fund_ids = fund_month_r.index
        fund_scale_r = fund_scale.loc[date, valid_fund_ids]

        fund_scale_r_win = fund_scale_r[fund_month_r < allocate_r]
        wr = 1 - fund_scale_r_win.sum() / fund_scale_r.sum()
        ranks.append(wr)

    print(np.mean(ranks))
    df_month_wr = pd.Series(data=ranks, index=allocate_month_inc.index)
    df_month_wr = df_month_wr.to_frame('wr')
    df_month_wr = df_month_wr[df_month_wr.index > '2012-12-31']
    df_month_wr.to_csv('data/df_month_wr.csv', index_label='date')

    return df_month_wr


def cal_quarter_wr(allocate_inc, df_inc):

    fund_month_inc = df_inc.resample('Q').sum()
    allocate_month_inc = allocate_inc.resample('Q').sum()

    ranks = []
    for date in allocate_month_inc.index:
        allocate_r = allocate_month_inc.loc[date]
        fund_month_r = fund_month_inc.loc[date].ravel()
        fund_month_r = list(fund_month_r[fund_month_r > 0.0])
        fund_month_r.append(allocate_r)
        fund_month_r.sort(reverse=True)
        # print(fund_month_r.index(allocate_r), len(fund_month_r), 1.0 * fund_month_r.index(allocate_r) / len(fund_month_r))
        ranks.append(1.0 * fund_month_r.index(allocate_r) / len(fund_month_r))
    print(np.mean(ranks))
    df_month_wr = pd.Series(data=ranks, index=allocate_month_inc.index)
    df_month_wr = df_month_wr.to_frame('wr')
    df_month_wr = df_month_wr[df_month_wr.index > '2012-12-31']
    df_month_wr.to_csv('data/df_quarter_wr.csv', index_label='date')

    return df_month_wr


def cal_quarter_money_wr(allocate_inc, df_inc, fund_scale):

    fund_month_inc = df_inc.resample('Q').sum()
    allocate_month_inc = allocate_inc.resample('Q').sum()
    fund_scale = fund_scale.resample('Q').last()

    ranks = []
    for date in allocate_month_inc.index:

        allocate_r = allocate_month_inc.loc[date]
        fund_month_r = fund_month_inc.loc[date]
        fund_month_r = fund_month_r.replace(0.0, np.nan).dropna()
        valid_fund_ids = fund_month_r.index
        fund_scale_r = fund_scale.loc[date, valid_fund_ids]

        fund_scale_r_win = fund_scale_r[fund_month_r < allocate_r]
        wr = 1 - fund_scale_r_win.sum() / fund_scale_r.sum()
        ranks.append(wr)

    print(np.mean(ranks))
    df_month_wr = pd.Series(data=ranks, index=allocate_month_inc.index)
    df_month_wr = df_month_wr.to_frame('wr')
    df_month_wr = df_month_wr[df_month_wr.index > '2012-12-31']
    df_month_wr.to_csv('data/df_quarter_wr.csv', index_label='date')

    return df_month_wr


def cal_halfyear_wr(allocate_inc, df_inc):

    fund_month_inc = df_inc.resample('6m', loffset='-1m').sum()
    allocate_month_inc = allocate_inc.resample('6m', loffset='-1m').sum()

    ranks = []
    for date in allocate_month_inc.index:
        allocate_r = allocate_month_inc.loc[date]
        fund_month_r = fund_month_inc.loc[date].ravel()
        fund_month_r = list(fund_month_r[fund_month_r > 0.0])
        fund_month_r.append(allocate_r)
        fund_month_r.sort(reverse=True)
        # print(fund_month_r.index(allocate_r), len(fund_month_r), 1.0 * fund_month_r.index(allocate_r) / len(fund_month_r))
        ranks.append(1.0 * fund_month_r.index(allocate_r) / len(fund_month_r))
    print(np.mean(ranks))
    df_month_wr = pd.Series(data=ranks, index=allocate_month_inc.index)
    df_month_wr = df_month_wr.to_frame('wr')
    df_month_wr = df_month_wr[df_month_wr.index > '2012-12-31']
    df_month_wr.to_csv('data/df_halfyear_wr.csv', index_label='date')

    return df_month_wr


def cal_halfyear_money_wr(allocate_inc, df_inc, fund_scale):

    fund_month_inc = df_inc.resample('6m', loffset='-1m').sum()
    allocate_month_inc = allocate_inc.resample('6m', loffset='-1m').sum()
    fund_scale = fund_scale.resample('6m', loffset='-2m').last()

    ranks = []
    for date in allocate_month_inc.index:

        allocate_r = allocate_month_inc.loc[date]
        fund_month_r = fund_month_inc.loc[date]
        fund_month_r = fund_month_r.replace(0.0, np.nan).dropna()
        valid_fund_ids = fund_month_r.index
        fund_scale_r = fund_scale.loc[date, valid_fund_ids]

        fund_scale_r_win = fund_scale_r[fund_month_r < allocate_r]
        wr = 1 - fund_scale_r_win.sum() / fund_scale_r.sum()
        ranks.append(wr)

    print(np.mean(ranks))
    df_month_wr = pd.Series(data=ranks, index=allocate_month_inc.index)
    df_month_wr = df_month_wr.to_frame('wr')
    df_month_wr = df_month_wr[df_month_wr.index > '2012-12-31']
    df_month_wr.to_csv('data/df_halfyear_wr.csv', index_label='date')

    return df_month_wr


def cal_year_wr(allocate_inc, df_inc):

    fund_month_inc = df_inc.resample('y').sum()
    allocate_month_inc = allocate_inc.resample('y').sum()

    ranks = []
    for date in allocate_month_inc.index:
        allocate_r = allocate_month_inc.loc[date]
        fund_month_r = fund_month_inc.loc[date].ravel()
        fund_month_r = list(fund_month_r[fund_month_r > 0.0])
        fund_month_r.append(allocate_r)
        fund_month_r.sort(reverse = True)
        # print(fund_month_r.index(allocate_r), len(fund_month_r), 1.0 * fund_month_r.index(allocate_r) / len(fund_month_r))
        ranks.append(1.0 * fund_month_r.index(allocate_r) / len(fund_month_r))
    print(np.mean(ranks))
    df_year_wr = pd.Series(data=ranks, index=allocate_month_inc.index)
    df_year_wr = df_year_wr.to_frame('wr')
    df_year_wr = df_year_wr[df_year_wr.index > '2012-12-31']
    df_year_wr.to_csv('data/df_year_wr.csv', index_label='date')

    return df_year_wr


def cal_year_money_wr(allocate_inc, df_inc, fund_scale):

    fund_month_inc = df_inc.resample('y').sum()
    allocate_month_inc = allocate_inc.resample('y').sum()
    fund_scale = fund_scale.resample('y').last()

    ranks = []
    for date in allocate_month_inc.index:

        allocate_r = allocate_month_inc.loc[date]
        fund_month_r = fund_month_inc.loc[date]
        fund_month_r = fund_month_r.replace(0.0, np.nan).dropna()
        valid_fund_ids = fund_month_r.index
        fund_scale_r = fund_scale.loc[date, valid_fund_ids]

        fund_scale_r_win = fund_scale_r[fund_month_r < allocate_r]
        wr = 1 - fund_scale_r_win.sum() / fund_scale_r.sum()
        ranks.append(wr)

    print(np.mean(ranks))
    df_month_wr = pd.Series(data=ranks, index=allocate_month_inc.index)
    df_month_wr = df_month_wr.to_frame('wr')
    df_month_wr = df_month_wr[df_month_wr.index > '2012-12-31']
    df_month_wr.to_csv('data/df_year_wr.csv', index_label='date')

    return df_month_wr
