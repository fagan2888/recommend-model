#coding=utf8


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


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func, literal_column
from tabulate import tabulate
from db import *
from util import xdict
from ipdb import set_trace
from asset import Asset

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def compare(ctx):
    '''
        compare multiple asset_allocation
    '''
    pass


@compare.command()
@click.option('--id1', 'optid1', default='MZ.000010', help=u'markowitz_id')
@click.option('--id2', 'optid2', default='MZ.000020', help=u'markowitz_id')
@click.option('--start-date', 'optstartdate', default='2012-07-27', help=u'markowitz startdate')
@click.option('--end-date', 'optenddate', default=None, help=u'markowitz endate')
@click.pass_context
def compare_markowitz(ctx, optid1, optid2, optstartdate, optenddate):
    print optid1, optid2

    nav1 = asset_mz_markowitz_nav.load_series(optid1, None, optstartdate, optenddate)
    nav2 = asset_mz_markowitz_nav.load_series(optid2, None, optstartdate, optenddate)

    nav1.name = optid1
    nav2.name = optid2

    df_nav = pd.concat([nav1, nav2], 1)
    df_ret = np.log(df_nav).diff().dropna()
    df_ret = df_ret.groupby(pd.Grouper(freq = 'M')).sum()
    df_ret = np.exp(df_ret)-1
    df_comp = df_ret[optid1] - df_ret[optid2]

    wr = len(df_comp[df_comp > 0]) / float(len(df_comp))
    print 'month win ratio:', wr


@compare.command()
@click.option('--id', 'optid', default='MZ.000010', help=u'markowitz_id')
@click.option('--ret', 'ret', default=0.01, help=u'target ret')
@click.option('--loss', 'loss', default=-0.01, help=u'target loss')
@click.option('--period', 'period', default=30, help=u'invest horizon')
@click.pass_context
def markowitz_wr(ctx, optid, ret = 0.01, loss = -0.01, period = 30):

    def target_test(nav):
        realized_loss = min(nav)/nav[0] - 1
        realized_ret = nav[-1]/nav[0] - 1
        if realized_loss < loss or realized_ret < ret:
            return 0
        else:
            return 1
    nav = asset_mz_markowitz_nav.load_series(optid, None)
    nav_wr = nav.rolling(period).apply(target_test).dropna()
    wr = nav_wr.sum()/len(nav_wr)

    print 'target_realized_ratio:', wr


@compare.command()
@click.option('--id1', 'secode1', default=2070000105, help=u'markowitz_id')
@click.option('--id2', 'secode2', default=2070000060, help=u'markowitz_id')
@click.option('--start-date', 'optstartdate', default='2005-01-04', help=u'markowitz startdate')
@click.option('--end-date', 'optenddate', default=None, help=u'markowitz endate')
@click.pass_context
def compare_caihui(ctx, secode1, secode2, optstartdate, optenddate):
    print secode1, secode2

    # nav1 = asset_mz_markowitz_nav.load_series(secode1, None, optstartdate, optenddate)
    # nav2 = asset_mz_markowitz_nav.load_series(secode2, None, optstartdate, optenddate)
    nav1 = caihui_tq_qt_index.load_index_nav(secode1)
    nav2 = caihui_tq_qt_index.load_index_nav(secode2)

    df_nav = pd.concat([nav1, nav2], 1)
    df_ret = np.log(df_nav).diff().dropna()
    # df_ret = df_ret.groupby(pd.Grouper(freq = 'M')).sum()
    # df_ret = df_ret.groupby(pd.Grouper(freq = 'W')).sum()
    df_ret = np.exp(df_ret)-1
    df_comp = df_ret.iloc[:,0] - df_ret.iloc[:,1]

    wr = len(df_comp[df_comp > 0]) / float(len(df_comp))
    print 'month win ratio:', wr
    for year in range(2006, 2019):
        print year, len(df_comp[df_comp>0].loc[str(year)])/float(len(df_comp.loc[str(year)]))
    set_trace()


@compare.command()
@click.option('--id', 'pool_id', default='94101', help=u'fund pool id')
@click.option('--type', 'pool_type', default='12', help=u'fund pool type')
@click.pass_context
def pool_test(ctx, pool_id, pool_type):

    pool_fund = asset_ra_pool_fund.load(11110101)
    pool_fund.index = pool_fund.index.droplevel(1)
    dates = pool_fund.index.unique()
    pre_change_list = None
    count = 0
    warning_funds_count = 0
    for pre_date, date in zip(dates[:-1], dates[1:]):

        pre_funds = pool_fund.loc[pre_date].ra_fund_code.values
        funds = pool_fund.loc[date].ra_fund_code.values
        change_list = []
        all_funds = np.union1d(pre_funds, funds)
        common_funds = np.intersect1d(pre_funds, funds)
        for fund in all_funds:
            if not fund in common_funds:
                change_list.append(fund)

        warning_list = np.intersect1d(pre_change_list, change_list)
        print date, len(warning_list)
        print warning_list
        count += 1.0
        warning_funds_count += len(warning_list)
        pre_change_list = change_list

    print 'warning_frequency:', warning_funds_count / count


