#coding=utf8


import getopt
import string
import json
import os
import sys
import logging
sys.path.append('shell')
import click
import config
import pandas as pd
import numpy as np
import LabelAsset
import os
import DBData
import time
import Const
import DFUtil
import LabelAsset
import Financial as fin
from ipdb import set_trace

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund, asset_stock_factor, asset_fund_factor, base_fund_fee, asset_ra_pool_fund
from asset import Asset, StockFundAsset
from asset_allocate import MonetaryAllocate
from trade_date import ATradeDate
from monetary_fund_filter import MonetaryFundFilter

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.option('--id', 'optid', default='12101,12201', help='reshape id')
@click.option('--datadir', '-d', type=click.Path(exists=True), help='dir used to store tmp data')
@click.pass_context
def pool(ctx, optid, datadir):
    '''fund pool group
    '''
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        ctx.invoke(fund, optid=optid, datadir=datadir)
        ctx.invoke(nav, optid=optid)
    else:
        # click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        pass


@pool.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), help='dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2010-01-08', help='start date to calc')
@click.option('--end-date', 'enddate', help='end date to calc')
@click.option('--period', 'optperiod', type=int, default=1, help='adjust period by month')
@click.option('--id', 'optid', help='fund pool id to update')
@click.option('--list/--no-list', 'optlist', default=False, help='list pool to update')
@click.option('--calc/--no-calc', 'optcalc', default=True, help='re calc label')
@click.option('--limit', 'optlimit', type=int, default=20, help='how many fund selected for each category')
@click.option('--eliminate_ratio', 'opteliminateratio', type=float, default=0.2, help='ratio to eliminate fund')
@click.option('--points', 'optpoints', help='Adjust points')
@click.pass_context
def fund(ctx, datadir, startdate, enddate, optid, optlist, optlimit, opteliminateratio, optcalc, optperiod, optpoints):
    '''run constant risk model
    '''
    if datadir is None:
        datadir = "./tmp"
    Const.datadir = datadir

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1));
        enddate = yesterday.strftime("%Y-%m-%d")
    if optid is not None:
        pools = [s.strip() for s in optid.split(',')]
    else:
        pools = None
    df_pool = load_pools(pools, [1, 2])

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_pool, headers='keys', tablefmt='psql'))
        return 0

    if optpoints is not None:
        adjust_points=pd.DatetimeIndex(optpoints.split(','))
    else:
        adjust_points = get_adjust_point(label_period=optperiod, startdate=startdate)

    print("adjust point:")
    for date in adjust_points:
        print(date.strftime("%Y-%m-%d"))

    for _, pool in df_pool.iterrows():
        fund_update(pool, adjust_points, optlimit, opteliminateratio, optcalc)


@pool.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), help='dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2010-01-08', help='start date to calc')
@click.option('--end-date', 'enddate', help='end date to calc')
@click.option('--period', 'optperiod', type=int, default=13, help='adjust period by week')
@click.option('--id', 'optid', help='fund pool id to update')
@click.option('--list/--no-list', 'optlist', default=False, help='list pool to update')
@click.option('--calc/--no-calc', 'optcalc', default=True, help='re calc label')
@click.option('--limit', 'optlimit', type=int, default=20, help='how many fund selected for each category')
@click.option('--points', 'optpoints', help='Adjust points')
@click.pass_context
def fund_corr_jensen(ctx, datadir, startdate, enddate, optid, optlist, optlimit, optcalc, optperiod, optpoints):
    '''run constant risk model
    '''
    if datadir is None:
        datadir = "./tmp"
    Const.datadir = datadir

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1));
        enddate = yesterday.strftime("%Y-%m-%d")
    if optid is not None:
        pools = [s.strip() for s in optid.split(',')]
    else:
        pools = None
    df_pool = load_pools(pools)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_pool, headers='keys', tablefmt='psql'))
        return 0

    if optpoints is not None:
        adjust_points=pd.DatetimeIndex(optpoints.split(','))
    else:
        adjust_points = get_adjust_point(label_period=optperiod, startdate=startdate)

    print("adjust point:")
    for date in adjust_points:
        print(date.strftime("%Y-%m-%d"))

    for _, pool in df_pool.iterrows():
        fund_update_corr_jensen(pool, adjust_points, optlimit, optcalc)


@pool.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), help='dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2011-02-01', help='start date to calc')
# @click.option('--end-date', 'enddate', default = '2018-05-31', help='end date to calc')
@click.option('--end-date', 'enddate', help='end date to calc')
@click.option('--period', 'optperiod', type=int, default=1, help='adjust period by week')
@click.option('--id', 'optid', help='fund pool id to update')
@click.option('--list/--no-list', 'optlist', default=False, help='list pool to update')
@click.option('--calc/--no-calc', 'optcalc', default=True, help='re calc label')
@click.option('--limit', 'optlimit', type=int, default=20, help='how many fund selected for each category')
@click.option('--points', 'optpoints', help='Adjust points')
@click.pass_context
def fund_factor_pool(ctx, datadir, startdate, enddate, optid, optlist, optlimit, optcalc, optperiod, optpoints):
    '''run constant risk model
    '''
    if datadir is None:
        datadir = "./tmp"
    Const.datadir = datadir

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1));
        enddate = yesterday.strftime("%Y-%m-%d")
    if optid is not None:
        pools = [s.strip() for s in optid.split(',')]
    else:
        pools = None
    df_pool = load_pools(pools)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_pool, headers='keys', tablefmt='psql'))
        return 0

    if optpoints is not None:
        adjust_points=pd.DatetimeIndex(optpoints.split(','))
    else:
        adjust_points = get_adjust_point(label_period=optperiod, startdate=startdate, enddate = enddate)

    print("adjust point:")
    for date in adjust_points:
        print(date.strftime("%Y-%m-%d"))

    for _, pool in df_pool.iterrows():
        fund_update_factor_pool(pool, adjust_points, optlimit, optcalc)


def fund_update_factor_pool(pool, adjust_points, optlimit, optcalc):
    ''' re calc fund for single fund pool
    '''
    lookback = pool.ra_lookback
    limit = optlimit

    if optcalc:
        #
        # 计算每个调仓点的最新配置
        #
        # ffr = asset_fund_factor.load_fund_factor_return()
        ffr = asset_stock_factor.load_stock_factor_return()
        ffr = ffr.unstack().T
        ffr.index = ffr.index.levels[1]
        ffe = asset_fund_factor.load_fund_factor_exposure()
        # ffe.to_csv('data/factor/fund_factor_exposure.csv', index_label = ['fund_id', 'ff_id', 'trade_date'])
        # ffe = pd.read_csv('data/factor/fund_factor_exposure.csv', index_col = ['fund_id', 'ff_id', 'trade_date'], parse_dates = ['trade_date'])
        ffe = ffe.reset_index()
        data = []
        pre_codes = None

        with click.progressbar(length=len(adjust_points), label='calc pool %s' % (pool.id)) as bar:
            for day in adjust_points:
                bar.update(1)
                # codes, codes_pool = pool_by_factor(pool, day, lookback, limit, ffr, ffe)
                codes, codes_pool = pool_by_single_factor(pool, day, lookback, limit, ffr, ffe)
                if pre_codes is None:
                    pre_codes = codes
                else:
                    pre_codes_origin = pre_codes.copy()
                    pre_codes = np.intersect1d(pre_codes, codes_pool)
                    new_codes = np.setdiff1d(codes, pre_codes)
                    fund_lack_num = limit - len(pre_codes)
                    codes = np.append(pre_codes, new_codes[:fund_lack_num])
                    pre_codes = codes
                    print('turnover rate:', 1 - len(np.intersect1d(codes, pre_codes_origin))/float(limit))

                print(day, len(codes), codes)
                if codes is None or len(codes) == 0:
                    continue
                ra_fund = base_ra_fund.load(codes = codes)
                ra_fund = ra_fund.set_index(['ra_code'])
                ra_pool = pool['id']
                for code in ra_fund.index:
                    ra_fund_id = ra_fund.loc[code, 'globalid']
                    data.append([ra_pool, day, ra_fund_id, code])
        fund_df = pd.DataFrame(data, columns = ['ra_pool', 'ra_date', 'ra_fund_id', 'ra_fund_code'])
        fund_df = fund_df.set_index(['ra_pool', 'ra_date', 'ra_fund_id'])

        df_new = fund_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        db = database.connection('asset')
        ra_pool_fund_t = Table('ra_pool_fund', MetaData(bind=db), autoload=True)
        s = s.where(ra_pool_fund_t.c.ra_pool.in_(df_new.index.get_level_values(0).tolist()))
        df_old = pd.read_sql(s, db, index_col = df_new.index.names)
        database.batch(db, ra_pool_fund_t, df_new, df_old)


@pool.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), help='dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2011-02-01', help='start date to calc')
@click.option('--end-date', 'enddate', help='end date to calc')
@click.option('--period', 'optperiod', type=int, default=1, help='adjust period by week')
@click.option('--id', 'optid', help='fund pool id to update')
@click.option('--list/--no-list', 'optlist', default=False, help='list pool to update')
@click.option('--calc/--no-calc', 'optcalc', default=True, help='re calc label')
@click.option('--limit', 'optlimit', type=int, default=2, help='how many fund selected for each category')
@click.option('--points', 'optpoints', help='Adjust points')
@click.pass_context
def fund_monetary_pool(ctx, datadir, startdate, enddate, optid, optlist, optlimit, optcalc, optperiod, optpoints):
    '''run constant risk model
    '''
    if datadir is None:
        datadir = "./tmp"
    Const.datadir = datadir

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1));
        enddate = yesterday.strftime("%Y-%m-%d")
    if optid is not None:
        pools = [s.strip() for s in optid.split(',')]
    else:
        pools = None
    df_pool = load_pools(pools)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_pool, headers='keys', tablefmt='psql'))
        return 0

    if optpoints is not None:
        adjust_points = pd.DatetimeIndex(optpoints.split(','))
    else:
        #adjust_points = get_week_adjust_point(label_period=optperiod, startdate=startdate, enddate=enddate)
        adjust_points = get_adjust_point(label_period=optperiod, startdate=startdate, enddate=enddate)

    print("adjust point:")
    for date in adjust_points:
        print(date.strftime("%Y-%m-%d"))

    for _, pool in df_pool.iterrows():
        if pool['ra_type'] == 1:
            fund_update_monetary_pool(pool, adjust_points, optlimit, optcalc)
        else:
            print('manual pool')


def fund_update_monetary_pool(pool, adjust_points, optlimit, optcalc):
    ''' re calc fund for single fund pool
    '''
    lookback = pool.ra_lookback
    limit = optlimit

    if optcalc:
        #
        # 计算每个调仓点的最新配置
        #
        data = []

        with click.progressbar(length=len(adjust_points), label='calc pool %s' % (pool.id)) as bar:
            for day in adjust_points:
                bar.update(1)
                mnf = MonetaryFundFilter()
                mnf.handle()
                asset_ids = MonetaryAllocate.all_monetary_fund_globalid()
                trade_dates = ATradeDate.week_trade_date()
                dict_nav = {}
                for asset_id in asset_ids:
                    dict_nav[asset_id] = Asset.load_nav_series(asset_id, reindex=trade_dates)
                df_nav = pd.DataFrame(dict_nav).fillna(method='pad')
                df_inc = df_nav.pct_change().fillna(0.0)

                codes = pool_by_scale_return(pool, day, lookback, limit, mnf, df_inc)

                print(day, len(codes), codes)
                if codes is None or len(codes) == 0:
                    continue
                ra_fund = base_ra_fund.load(globalids=codes)
                ra_fund = ra_fund.set_index(['ra_code'])
                ra_pool = pool['id']
                for code in ra_fund.index:
                    ra_fund_id = ra_fund.loc[code, 'globalid']
                    data.append([ra_pool, day, ra_fund_id, code])

        fund_df = pd.DataFrame(data, columns = ['ra_pool', 'ra_date', 'ra_fund_id', 'ra_fund_code'])
        fund_df = fund_df.set_index(['ra_pool', 'ra_date', 'ra_fund_id'])

        df_new = fund_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        db = database.connection('asset')
        ra_pool_fund_t = Table('ra_pool_fund', MetaData(bind=db), autoload=True)
        s = s.where(ra_pool_fund_t.c.ra_pool.in_(df_new.index.get_level_values(0).tolist()))
        df_old = pd.read_sql(s, db, index_col = df_new.index.names)
        database.batch(db, ra_pool_fund_t, df_new, df_old)


def fund_update_corr_jensen(pool, adjust_points, optlimit, optcalc):
    ''' re calc fund for single fund pool
    '''
    lookback = pool.ra_lookback
    limit = optlimit

    if optcalc:
        #
        # 计算每个调仓点的最新配置
        #

        db = database.connection('asset')
        ra_pool_sample_t = Table('ra_pool_sample', MetaData(bind=db), autoload=True)
        ra_pool_fund_t= Table('ra_pool_fund', MetaData(bind=db), autoload=True)

        data = []
        with click.progressbar(length=len(adjust_points), label='calc pool %s' % (pool.id)) as bar:
            for day in adjust_points:
                bar.update(1)
                codes = pool_by_corr_jensen(pool, day, lookback, limit)
                print(day, codes)
                if codes is None or len(codes) == 0:
                    continue
                ra_fund = base_ra_fund.load(codes = codes)
                ra_fund = ra_fund.set_index(['ra_code'])
                ra_pool    = pool['id']
                for code in ra_fund.index:
                    ra_fund_id = ra_fund.loc[code, 'globalid']
                    data.append([ra_pool, day, ra_fund_id, code])
        fund_df = pd.DataFrame(data, columns = ['ra_pool', 'ra_date', 'ra_fund_id', 'ra_fund_code'])
        fund_df = fund_df.set_index(['ra_pool', 'ra_date', 'ra_fund_id'])

        df_new = fund_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        s = s.where(ra_pool_fund_t.c.ra_pool.in_(df_new.index.get_level_values(0).tolist()))
        df_old = pd.read_sql(s, db, index_col = df_new.index.names)
        database.batch(db, ra_pool_fund_t, df_new, df_old)


def fund_lowliest_elimination(df_pre_fund, df_indicator, df_label, ratio, limit):

    #categories = ['largecap','smallcap','rise','decline','oscillation','growth','value']
    #print df_label.head()
    #set_trace()
    data = {}
    for category in df_label.columns:
        index_codes = df_label[df_label[category] == 1].index
        df_tmp = df_indicator.loc[index_codes]
        if df_pre_fund is None:
            data[category] = df_tmp.sort_values(by='jensen', ascending=False)[0:limit]
        else:
            pre_fund = df_indicator.loc[df_pre_fund.loc[category].index]
            eliminate_num = int(ratio * len(pre_fund))
            pre_fund = pre_fund.sort_values(by='jensen', ascending=False)[0:len(pre_fund) - eliminate_num]
            df_tmp = df_tmp.loc[df_tmp.index.difference(pre_fund.index)]
            df_tmp = df_tmp.sort_values(by='jensen', ascending=False)
            df_tmp = pd.concat([pre_fund, df_tmp], axis = 0)
            #print pre_fund
            #print df_tmp[0:limit]
            data[category] = df_tmp[0:limit]

    df_result = pd.concat(data, names=['category','code'])

    return df_result


def fund_update(pool, adjust_points, optlimit, opteliminateratio, optcalc):
    ''' re calc fund for single fund pool
    '''
    lookback = pool.ra_lookback
    limit = optlimit

    if optcalc:
        #
        # 计算每个调仓点的最新配置
        #
        data_fund = {}
        with click.progressbar(length=len(adjust_points), label=('calc pool %s' % (pool.id)).ljust(30)) as bar:
            for day in adjust_points:
                #print day
                bar.update(1)
                if pool['ra_fund_type'] == 1:
                    df_indicator, df_label = LabelAsset.label_asset_stock_per_day(day, lookback, limit)
                else:
                    df_indicator, df_label = LabelAsset.label_asset_bond_per_day(day, lookback, limit)

                fund_dates = np.array(list(data_fund.keys()))
                fund_date = fund_dates[fund_dates < day]
                fund_date = list(fund_date)
                fund_date.sort()
                if len(fund_date) == 0:
                    data_fund[day] = fund_lowliest_elimination(None, df_indicator, df_label, opteliminateratio, optlimit)
                else:
                    data_fund[day] = fund_lowliest_elimination(data_fund[fund_date[-1]], df_indicator, df_label, opteliminateratio, optlimit)


        df_fund = pd.concat(data_fund, names=['ra_date', 'ra_category', 'ra_fund_code'])

        df_new = df_fund.rename(index=DFUtil.categories_types(True), columns={'date':'ra_date', 'category':'ra_category', 'code':'ra_fund_code', 'sharpe':'ra_sharpe',  'jensen':'ra_jensen', 'sortino':'ra_sortino', 'ppw':'ra_ppw'})
        df_new.drop('stability', axis=1, inplace=True)

        df_new = df_new.applymap(lambda x: round(x, 4) if type(x) == float else x)

        codes = df_new.index.get_level_values(2)
        xtab = fund_code_to_globalid(codes)
        df_new['ra_fund_id'] = xtab[df_new.index.get_level_values('ra_fund_code')].values
        df_new['ra_pool'] = pool.id
        df_new['ra_fund_type'] = 1
        df_new['ra_fund_level'] = 1

        df_new.reset_index(inplace=True)
        df_new = df_new.reindex_axis(['ra_pool', 'ra_category',  'ra_date', 'ra_fund_id', 'ra_fund_code', 'ra_fund_type', 'ra_fund_level', 'ra_sharpe', 'ra_jensen', 'ra_sortino', 'ra_ppw'], axis='columns')
        df_new.sort_values(by=['ra_pool', 'ra_category', 'ra_date', 'ra_fund_id'], inplace=True)

        #df_new.to_csv(datapath('pool_%s.csv' % (pool['id'])), index=False)

    else:
        df_new = pd.read_csv(datapath('pool_%s.csv' % (pool['id'])), parse_dates=['ra_date'], dtype={'ra_fund_code': str})

    df_new.set_index(['ra_pool', 'ra_category', 'ra_date', 'ra_fund_id'], inplace=True)
    df_new = df_new.applymap(lambda x: '%.4f' % (x) if type(x) == float else x)

    db = database.connection('asset')
    ra_pool_fund = Table('ra_pool_fund', MetaData(bind=db), autoload=True)

    # 加载就数据
    columns2 = [
        ra_pool_fund.c.ra_pool,
        ra_pool_fund.c.ra_category,
        ra_pool_fund.c.ra_date,
        ra_pool_fund.c.ra_fund_id,
        ra_pool_fund.c.ra_fund_code,
        ra_pool_fund.c.ra_fund_type,
        ra_pool_fund.c.ra_fund_level,
        ra_pool_fund.c.ra_sharpe,
        ra_pool_fund.c.ra_jensen,
        ra_pool_fund.c.ra_sortino,
        ra_pool_fund.c.ra_ppw,
    ]
    stmt_select = select(columns2, ra_pool_fund.c.ra_pool == pool.id)
    df_old = pd.read_sql(stmt_select, db, index_col=['ra_pool', 'ra_category', 'ra_date', 'ra_fund_id'])
    if not df_old.empty:
        df_old = df_old.applymap(lambda x: '%.4f' % (x) if type(x) == float else x)

    database.batch(db, ra_pool_fund, df_new, df_old)


def fund_code_to_globalid(codes):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    ra_fund = Table('ra_fund', metadata, autoload=True)

    columns = [
        ra_fund.c.globalid,
        ra_fund.c.ra_code,
    ]
    if codes is not None:
        s = select(columns, ra_fund.c.ra_code.in_(codes))
    else:
        s = select(columns)

    df_result = pd.read_sql(s, db, index_col='ra_code')

    return df_result['globalid']


def load_pools(pools, pool_type=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    ra_pool = Table('ra_pool', metadata, autoload=True)

    columns = [
        ra_pool.c.id,
        ra_pool.c.ra_type,
        ra_pool.c.ra_algo,
        ra_pool.c.ra_date_type,
        ra_pool.c.ra_fund_type,
        ra_pool.c.ra_lookback,
        ra_pool.c.ra_index_id,
        ra_pool.c.ra_name,
    ]

    s = select(columns).where(ra_pool.c.ra_type != -1)
    if pools is not None:
        s = s.where(ra_pool.c.id.in_(pools))
    if pool_type is not None:
        if hasattr(pool_type, "__iter__") and not isinstance(pool_type, str):
            s = s.where(ra_pool.c.ra_fund_type.in_(pool_type))
        else:
            s = s.where(ra_pool.c.ra_fund_type == pool_type)

    df_pool = pd.read_sql(s, db)

    return df_pool


def get_adjust_point(startdate = '2010-01-08', enddate=None, label_period=1):
    # 加载时间轴数据
    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1));
        enddate = yesterday.strftime("%Y-%m-%d")

    index = base_trade_dates.load_trade_dates(startdate, end_date=enddate)
    index = index[index.td_type >= 8].index
    #index = ATradeDate.week_trade_date(startdate, enddate)
    #print(index)
    #index = DBData.trade_date_index(startdate, end_date=enddate)

    if label_period > 1:
        label_index = index[::label_period]
        if index.max() not in label_index:
            label_index = label_index.insert(len(label_index), index.max())
    else:
        label_index = index
    # label_index = pd.DatetimeIndex([
    #     '2010-01-08',
    #     '2010-07-09',
    #     '2011-01-07',
    #     '2011-07-08',
    #     '2012-01-13',
    #     '2012-07-20',
    #     '2013-01-25',
    #     '2013-08-02',
    #     '2014-01-30',
    #     '2014-08-01',
    #     '2015-01-30',
    #     '2015-07-31',
    #     '2016-01-29',
    #     '2016-08-05',
    #     '2016-11-05',
    # ])

    return label_index

def get_week_adjust_point(startdate = '2010-01-08', enddate=None, label_period=1):
    # 加载时间轴数据
    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1));
        enddate = yesterday.strftime("%Y-%m-%d")

    index = ATradeDate.week_trade_date(startdate, enddate)

    if label_period > 1:
        label_index = index[::label_period]
        if index.max() not in label_index:
            label_index = label_index.insert(len(label_index), index.max())
    else:
        label_index = index
    return label_index


@pool.command()
@click.option('--id', 'optid', help='ids of fund pool to update')
@click.option('--list/--no-list', 'optlist', default=False, help='list pool to update')
@click.pass_context
def nav(ctx, optid, optlist):
    ''' calc pool nav and inc
    '''
    db_asset = create_engine(config.db_asset_uri)
    # db_asset.echo = True
    db_base = create_engine(config.db_base_uri)
    db = {'asset':db_asset, 'base':db_base}

    if optid is not None:
        pools = [s.strip() for s in optid.split(',')]
    else:
        pools = None

    df_pool = load_pools(pools)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_pool, headers='keys', tablefmt='psql'))
        return 0

    for _, pool in df_pool.iterrows():
        nav_update(db, pool)

def nav_update(db, pool):
    df_categories = load_pool_category(pool['id'])
    categories = df_categories['ra_category']
    with click.progressbar(length=len(categories) + 1, label=('update nav for pool %s' % (pool.id)).ljust(30)) as bar:
        for category in categories:
            nav_update_category(db['asset'], pool, category)
            bar.update(1)
        else:
            nav_update_category(db['asset'], pool, 0)
            bar.update(1)
    #nav_update_category(db['asset'], pool, 0)

def nav_update_category(db, pool, category):
    # 加载基金列表
    df = load_fund_category(pool['id'], category)

    # 构建均分仓位
    df['ra_ratio'] = 1.0
    df.set_index('ra_fund_code', append=True, inplace=True)
    df['ra_ratio'] = df['ra_ratio'].groupby(level=0, group_keys=False).apply(lambda x: x / len(x))
    df_position = df.unstack().fillna(0.0)
    df_position.columns = df_position.columns.droplevel(0)


    # 加载基金收益率
    min_date = df_position.index.min()
    #max_date = df_position.index.max()
    max_date = (datetime.now() - timedelta(days=1)) # yesterday


    df_nav = DBData.db_fund_value_daily(
        min_date, max_date, codes=df_position.columns)
    if '000000' in df_position.columns:
        df_nav['000000'] = 1
    df_inc = df_nav.pct_change().fillna(0.0)

    # 计算复合资产净值
    df_nav_portfolio = DFUtil.portfolio_nav(df_inc, df_position, result_col='portfolio')
    # df_nav_portfolio.to_csv(datapath('category_nav_' + category + '.csv'))

    df_result = df_nav_portfolio[['portfolio']].rename(columns={'portfolio':'ra_nav'}).copy()
    df_result.index.name = 'ra_date'
    df_result['ra_inc'] = df_result['ra_nav'].pct_change().fillna(0.0)
    df_result['ra_pool'] = pool['id']
    df_result['ra_category'] = category
    df_result['ra_type'] = pool['ra_type']
    df_result = df_result.reset_index().set_index(['ra_pool', 'ra_category', 'ra_type', 'ra_date'])

    df_new = df_result.apply(format_nav_and_inc)


    # 加载旧数据
    t2 = Table('ra_pool_nav', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.ra_pool,
        t2.c.ra_category,
        t2.c.ra_type,
        t2.c.ra_date,
        t2.c.ra_nav,
        t2.c.ra_inc,
    ]
    stmt_select = select(columns2, (t2.c.ra_pool == pool['id']) & (t2.c.ra_category == category) & (t2.c.ra_type == pool['ra_type']))
    df_old = pd.read_sql(stmt_select, db, index_col=['ra_pool', 'ra_category', 'ra_type', 'ra_date'], parse_dates=['ra_date'])
    if not df_old.empty:
        df_old = df_old.apply(format_nav_and_inc)

    # 更新数据库
    database.batch(db, t2, df_new, df_old, timestamp=False)

def format_nav_and_inc(x):
    if x.name == "ra_nav":
        ret = x.map("{:.6f}".format)
    elif x.name == "ra_inc":
        ret = x.map("{:.6f}".format)
    else:
        ret = x

    return ret


def load_pool_category(pid):
    db = database.connection('asset')
    t = Table('ra_pool_fund', MetaData(bind=db), autoload=True)

    # 加载就数据
    columns = [
        t.c.ra_pool,
        t.c.ra_category,
    ]
    stmt_select = select(columns, t.c.ra_pool == pid).distinct()

    df = pd.read_sql(stmt_select, db)

    return df

def load_fund_category(pid, category):
    db = database.connection('asset')
    # 加载基金列表
    t = Table('ra_pool_fund', MetaData(bind=db), autoload=True)
    columns = [
        t.c.ra_date,
        t.c.ra_fund_code,
    ]
    s = select(columns, (t.c.ra_pool == pid))
    if category == 0:
        s = s.distinct()
    else:
        s = s.where(t.c.ra_category == category)

    df = pd.read_sql(s, db, index_col = ['ra_date'], parse_dates=['ra_date'])

    return df


@pool.command(name='import')
@click.option('--id', 'optid', type=int, help='specify fund pool id')
@click.option('--name', 'optname', help='specify fund pool name')
@click.option('--otype', 'optotype', type=click.Choice(['1', '9']), default='1', help='online type(1:expriment; 9:online)')
@click.option('--dtype', 'optdtype', type=click.Choice(['1', '2']), default='2', help='date type(1:day, 2:week)')
@click.option('--ftype', 'optftype', type=click.Choice(['0', '1', '2', '3', '4']), default='0', help='fund type(0:unknow; 1:stock; 2:bond; 3:money; 4:other)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help='replace pool if exists')
@click.argument('csv', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=False), required=True)
@click.pass_context
def import_command(ctx, csv, optid, optname, optotype, optdtype, optftype, optreplace):
    '''
    import fund pool from csv file
    '''

    #
    # 处理id参数
    #
    if optid is not None:
        #
        # 检查id是否存在
        #
        df_existed = load_pools([str(optid)])
        if not df_existed.empty:
            if optreplace:
                click.echo(click.style("allocation instance [optId] existed, will replace!", fg="yellow"))
            else:
                click.echo(click.style("allocation instance [optId] existed, import aborted!", fg="red"))
            return -1;
    else:
        #
        # 自动生成id
        #
        prefix = optotype + optdtype + optftype
        between_min, between_max = ('%s00' % (prefix), '%s99' % (prefix))

        max_id = query_max_pool_id_between(between_min, between_max)
        if max_id is None:
            optid = between_min;
        else:
            if max_id >= between_max:
                click.echo(click.style("run out of instance id [maxId], import aborted!", fg="red"))
                return -1

            if optreplace:
                optid = max_id
            else:
                optid = max_id + 1;

    #
    # 处理name参数
    #
    if optname is None:
        optname = os.path.basename(csv);


    db = database.connection('asset')
    metadata = MetaData(bind=db)
    ra_pool = Table('ra_pool', metadata, autoload=True)
    ra_pool_fund = Table('ra_pool_fund', metadata, autoload=True)

    #
    # 处理替换
    #
    if optreplace:
        ra_pool.delete(ra_pool.c.id == optid).execute()
        ra_pool_fund.delete(ra_pool_fund.c.ra_pool_id == optid).execute()

    now = datetime.now()
    #
    # 导入数据
    #
    row = {
        'id': optid, 'ra_type':optotype, 'ra_date_type':optdtype
        , 'ra_fund_type':optftype, 'ra_name': optname
        , 'created_at': func.now(), 'updated_at': func.now()
    }
    ra_pool.insert(row).execute()

    df = pd.read_csv(csv, parse_dates=['date'], dtype={'code':str})
    df = df.replace({'category':DFUtil.categories_types()})
    df = df.rename(columns={'date':'ra_date', 'category':'ra_category', 'code':'ra_fund_code', 'sharpe':'ra_sharpe',  'jensen':'ra_jensen', 'sortino':'ra_sortino', 'ppw':'ra_ppw', 'level':'ra_fund_level'})

    if 'ra_fund_level' not in df.columns:
        df['ra_fund_level'] = 1

    if 'ra_jensen' not in df.columns:
        df['ra_jensen'] = 0.0
    else:
        df['ra_jensen'] = np.round(df['ra_jensen'])

    if 'ra_sharpe' not in df.columns:
        df['ra_sharpe'] = 0.0
    else:
        df['ra_sharpe'] = np.round(df['ra_sharpe'])

    if 'ra_ppw' not in df.columns:
        df['ra_ppw'] = 0.0
    else:
        df['ra_ppw'] = np.round(df['ra_ppw'])

    if 'ra_sortino' not in df.columns:
        df['ra_sortino'] = 0.0
    else:
        df['ra_sortino'] = np.round(df['ra_sortino'])

    if 'stability' in df.columns:
        df =  df.drop('stability', axis = 1)


    #
    # 翻译基金代码 => 基金ID
    #
    df_tosave = DFUtil.merge_column_for_fund_id_type(df, 'ra_fund_code')
    df_tosave = df_tosave.rename(columns={'globalid':'ra_fund_id', 'ra_type':'ra_fund_type'})
    df_tosave['ra_pool'] = optid
    df_tosave.set_index(['ra_pool', 'ra_category', 'ra_date', 'ra_fund_id'], inplace=True)

    df_tosave['updated_at'] = df_tosave['created_at'] = now

    df_tosave.to_sql(ra_pool_fund.name, db, index=True, if_exists='append')

    if len(df_tosave.index) > 1:
        logger.info("insert %s (%5d) : %s " % (ra_pool_fund.name, len(df_tosave.index), df_tosave.index[0]))

    click.echo(click.style("import complement! instance id [%s]" % (optid), fg='green'))


    return 0

def query_max_pool_id_between(min_id, max_id):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('ra_pool', metadata, autoload=True)

    columns = [ t.c.id ]

    s = select([func.max(t.c.id).label('maxid')]).where(t.c.id.between(min_id, max_id))

    return s.execute().scalar()

@pool.command()
@click.option('--id', 'optid', help='specify fund pool id (e.g. 12101,92101')
@click.option('--list/--no-list', 'optlist', default=False, help='list pool to update')
@click.pass_context
def turnover(ctx, optid, optlist):
    ''' calc pool turnover
    '''
    if optid is not None:
        pools = [s.strip() for s in optid.split(',')]
    else:
        pools = None
    df_pool = load_pools(pools)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_pool, headers='keys', tablefmt='psql'))
        return 0

    for _, pool in df_pool.iterrows():
        turnover_update(pool)

def turnover_update(pool):
    df_categories = load_pool_category(pool['id'])
    categories = df_categories['ra_category']

    with click.progressbar(length=len(categories) + 1, label=('update turnover for pool %s' % (pool.id)).ljust(30)) as bar:
        for category in categories:
            turnover_update_category(pool, category)
            bar.update(1)
        else:
            turnover_update_category(pool, 0)
            bar.update(1)

def turnover_update_category(pool, category):
    # 加载基金列表
    df = load_fund_category(pool['id'], category)

    # 构建均分仓位
    df['mask'] = 1
    df.set_index('ra_fund_code', append=True, inplace=True)
    df = df.unstack(fill_value=0)
    df_prev = df.shift(1).fillna(0).astype(int)

    df_prev.iloc[0] = df.iloc[0]

    df_and = np.bitwise_and(df, df_prev)

    df_new = (1 - df_and.sum(axis=1) / df_prev.sum(axis=1)).to_frame('ra_turnover')
    df_new['ra_pool'] = pool['id']
    df_new['ra_category'] = category

    df_new = df_new.reset_index().set_index(['ra_pool', 'ra_category', 'ra_date'])

    df_new = df_new.applymap("{:.4f}".format)


    db = database.connection('asset')
    # 加载旧数据
    t2 = Table('ra_pool_criteria', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.ra_pool,
        t2.c.ra_category,
        t2.c.ra_date,
        t2.c.ra_turnover,
    ]
    stmt_select = select(columns2, (t2.c.ra_pool == pool['id']) & (t2.c.ra_category == category))
    df_old = pd.read_sql(stmt_select, db, index_col=['ra_pool', 'ra_category', 'ra_date'], parse_dates=['ra_date'])
    if not df_old.empty:
        df_old = df_old.applymap("{:.4f}".format)

    # 更新数据库
    database.batch(db, t2, df_new, df_old, timestamp=False)

@pool.command()
@click.option('--id', 'optid', help='specify fund pool id (e.g. 12101,92101')
@click.option('--list/--no-list', 'optlist', default=False, help='list pool to update')
@click.pass_context
def corr(ctx, optid, optlist):
    ''' calc pool corr
    '''
    if optid is not None:
        pools = [s.strip() for s in optid.split(',')]
    else:
        pools = None

    df_pool = load_pools(pools)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_pool, headers='keys', tablefmt='psql'))
        return 0

    for _, pool in df_pool.iterrows():
        corr_update(pool)

def corr_update(pool):
    df_categories = load_pool_category(pool['id'])
    categories = df_categories['ra_category']

    with click.progressbar(length=len(categories) + 1, label=('update corr for pool %s' % (pool.id)).ljust(30)) as bar:
        for category in categories:
            corr_update_category(pool, category, 52)
            bar.update(1)
        else:
            corr_update_category(pool, 0, 52)
            bar.update(1)

def corr_update_category(pool, category, lookback):
    # 加载基金列表
    df = load_fund_category(pool['id'], category)

    data = {}
    for k0, v0 in df.groupby(level=0):
        index = DBData.trade_date_lookback_index(end_date=k0, lookback=lookback)
        start_date = index.min().strftime("%Y-%m-%d")
        end_date = k0

        df_nav = DBData.db_fund_value_daily(start_date, end_date, codes=v0['ra_fund_code'])
        df_inc = df_nav.pct_change().fillna(0.0)
        df_corr = df_inc.corr()
        df_corr.fillna(0.0, inplace=True)

        if df_corr.empty:
            data[k0] = 0.0
        else:
            data[k0] = np.mean(df_corr.values)

    df_new = pd.DataFrame({'ra_corr': data})
    df_new.index.name = 'ra_date'
    df_new['ra_pool'] = pool['id']
    df_new['ra_category'] = category

    df_new = df_new.reset_index().set_index(['ra_pool', 'ra_category', 'ra_date'])

    df_new = df_new.applymap("{:.4f}".format)


    db = database.connection('asset')
    # 加载旧数据
    t2 = Table('ra_pool_criteria', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.ra_pool,
        t2.c.ra_category,
        t2.c.ra_date,
        t2.c.ra_corr,
    ]
    stmt_select = select(columns2, (t2.c.ra_pool == pool['id']) & (t2.c.ra_category == category))
    df_old = pd.read_sql(stmt_select, db, index_col=['ra_pool', 'ra_category', 'ra_date'], parse_dates=['ra_date'])
    if not df_old.empty:
        df_old = df_old.applymap("{:.4f}".format)

    # 更新数据库
    database.batch(db, t2, df_new, df_old, timestamp=False)


def pool_by_corr_jensen(pool, day, lookback, limit):

    index = base_trade_dates.trade_date_lookback_index(end_date=day, lookback=lookback)

    start_date = index.min().strftime("%Y-%m-%d")
    end_date = day.strftime("%Y-%m-%d")


    ra_index_id = pool['ra_index_id']
    pool_id     = pool['id']
    df_nav_index = base_ra_index_nav.index_value(start_date, end_date, ra_index_id)
    #df_nav_index.index.name = str(df_nav_index.index.name)
    df_nav_index.columns = df_nav_index.columns.astype(str)
    if len(df_nav_index.index) == 0:
        return []
    pool_codes   = asset_ra_pool_sample.load(pool_id)['ra_fund_code'].values

    df_nav_fund  = base_ra_fund_nav.load_daily(start_date, end_date, codes = pool_codes)
    if len(df_nav_fund) == 0:
        return []

    df_nav_fund  = df_nav_fund.reindex(pd.date_range(df_nav_index.index[0], df_nav_index.index[-1]))
    df_nav_fund  = df_nav_fund.fillna(method = 'pad')
    df_nav_fund  = df_nav_fund.dropna(axis = 1)
    df_nav_fund  = df_nav_fund.loc[df_nav_index.index]
    fund_index_df = pd.concat([df_nav_index, df_nav_fund], axis = 1, join_axes = [df_nav_index.index])


    fund_index_corr_df = fund_index_df.pct_change().fillna(0.0).corr().fillna(0.0)

    corr = fund_index_corr_df[ra_index_id][1:]
    corr = corr.sort_values(ascending = False)

    code_jensen = {}
    for code in df_nav_fund.columns:
        jensen = fin.jensen(df_nav_fund[code].pct_change().fillna(0.0), df_nav_index[ra_index_id].pct_change().fillna(0.0) ,Const.rf)
        code_jensen.setdefault(code, jensen)


    if len(code_jensen) == 0:
        logger.info('No FUND')
        return None
    else:
        final_codes = []
        x = code_jensen
        sorted_x = sorted(iter(x.items()), key=lambda x : x[1], reverse=True)
        corr_threshold = np.percentile(corr.values, 80)
        corr_threshold = 0.7 if corr_threshold <= 0.7 else corr_threshold
        corr_threshold = 0.9 if corr_threshold >= 0.9 else corr_threshold
        for i in range(0, len(sorted_x)):
            code, jensen = sorted_x[i]
            if corr[code] >= corr_threshold:
                final_codes.append(code)

        final_codes = final_codes[0 : limit]
        return final_codes


def pool_by_single_factor(pool, day, lookback, limit, ffr, ffe):

    index = ATradeDate.week_trade_date(begin_date = day, end_date = day, lookback = lookback)
    sdate = index.min()
    edate = index.max()

    ffe = ffe[(ffe.trade_date >= sdate) & (ffe.trade_date <= edate)]
    ffe = ffe[['fund_id', 'ff_id', 'exposure']].groupby(['fund_id', 'ff_id']).last().unstack()
    ffe.columns = ffe.columns.levels[1]
    ffe = ffe.dropna()

    vf = np.zeros(ffe.shape[1])
    # pos = int(pool.ra_index_id[-2:])-1
    pos = ffe.columns.values.tolist().index(pool.ra_index_id)
    pool_type = pool.ra_fund_type // 100
    if pool_type == 114:
        vf[pos] = -1
    elif pool_type == 115:
        vf[pos] = 1
    elif pool_type == 113:
        vf[pos] = 1

    all_fund_scale = StockFundAsset.all_fund_scale()
    df_valid = all_fund_scale.loc[day].dropna()
    df_valid = df_valid[df_valid > 2e8].index
    df_valid = df_valid.intersection(ffe.index)

    score = np.dot(ffe, vf)
    df_score = pd.Series(score, index = ffe.index)
    df_score = df_score.loc[df_valid]
    df_score = df_score.sort_values(ascending = False)
    final_codes = df_score.index[:limit]
    final_codes_large = df_score.index[:5*limit]

    return final_codes, final_codes_large


def pool_by_scale_return(pool, day, lookback, limit, mnf, df_inc):

    fund_status = mnf.fund_status
    fund_status = fund_status[fund_status.fi_yingmi_amount <= 1e3]
    fund_status = fund_status[fund_status.fi_yingmi_subscribe_status == 0.0]
    #valid_ids_1 = fund_status.index
    valid_ids = fund_status.index


    fund_fee = mnf.fund_fee.ff_fee
    valid_ids_2 = fund_fee[fund_fee >= 0.00].index
    #valid_ids_2 = ['%d' % fund_code for fund_code in valid_ids_2]
    #valid_ids = np.intersect1d(valid_ids_1, valid_ids_2)

    #print(valid_ids_1)
    #print(valid_ids_2)
    #print('30002562' in valid_ids.ravel())
    #tmp_scale = mnf.fund_scale.loc[day]
    #tmp_scale = tmp_scale.sort_values(ascending=False)
    #scale_filter_codes = tmp_scale[tmp_scale > 1e8].index
    #scale_filter_codes = tmp_scale.index
    #scale_filter_ids = [str(mnf.fund_id_dict[fund_code]) for fund_code in scale_filter_codes]
    #print('33009272' in scale_filter_ids)

    #final_filter_ids = np.intersect1d(scale_filter_ids, valid_ids)
    #final_filter_ids = np.intersect1d(valid_ids, valid_ids)
    tmp_df_inc = df_inc.copy()
    final_filter_ids = valid_ids & tmp_df_inc.columns
    tmp_df_inc = tmp_df_inc[tmp_df_inc.index < day]
    tmp_df_inc = tmp_df_inc.iloc[-lookback:]
    tmp_df_inc = tmp_df_inc[final_filter_ids]
    #print(tmp_df_inc.columns)
    #print(final_filter_ids)

    #trade_dates = ATradeDate.week_trade_date()
    #dict_nav = {}
    #for asset_id in tmp_df_inc.columns:
    #    dict_nav[asset_id] = Asset.load_nav_series(asset_id, reindex=trade_dates)
    #df_nav = pd.DataFrame(dict_nav).fillna(method='pad')
    #df_inc = df_nav.pct_change().fillna(0.0)
    #df_inc = df_inc[df_inc.index > day]
    #df_inc = df_inc.iloc[0:4]
    #rs = df_inc.mean()
    #rs = rs.sort_values(ascending=False)
    #num = limit
    #pool_codes = []
    #for fund_globalid in rs.index:
    #        pool_codes.append(fund_globalid)
    #        if len(pool_codes) >= num:
    #            return pool_codes
    #qieman_funds = asset_ra_pool_fund.load(11310106)
    #qieman_funds = qieman_funds[qieman_funds.index.get_level_values(0) <= day]
    #qieman_funds = qieman_funds.tail(3).index.get_level_values(1).astype(str).ravel()

    fund_fee.index = fund_fee.index.astype(str)
    pool_codes = []
    num = limit
    rs = tmp_df_inc.mean()
    #a = fund_fee.loc[rs.index]
    #a.to_csv('fee.csv')
    #print('----', len(rs))
    rs = rs[rs > 0]
    #print('****', len(rs))
    #all_df_inc = all_df_inc.mean()
    rs = rs.sort_values(ascending=False)
    #if len(qieman_funds) == 0:
    #    threshold = rs.iloc[int(len(rs) * 0.15)]
    #else:
    #    new_qieman_funds = []
    #    for f in qieman_funds:
    #        if f in all_df_inc.index:
    #            new_qieman_funds.append(f)
    #    if len(new_qieman_funds) > 0:
    #        threshold = all_df_inc.loc[new_qieman_funds].mean() * 0.95
    #    else:
    #        threshold = rs.iloc[int(len(rs) * 0.15)]

    threshold = rs.iloc[int(len(rs) * 0.05)]
    rs = rs[rs >= threshold]
    #for fund_globalid in rs.index:
    #        pool_codes.append(fund_globalid)
    #        if len(pool_codes) >= num:
    #            return pool_codes

    #print(' fund num ', len(rs))
    for fund_globalid in rs.index:
        if int(fund_globalid) in valid_ids_2:
            #print('fee')
            #print(fund_globalid, fund_fee.loc[fund_globalid])
            pool_codes.append(fund_globalid)
        if len(pool_codes) >= num:
            return pool_codes
    for fund_globalid in rs.index:
        if fund_globalid not in pool_codes:
            #print('fee')
            #print(fund_globalid, fund_fee.loc[fund_globalid])
            pool_codes.append(fund_globalid)
        if len(pool_codes) >= num:
            return pool_codes
    return pool_codes
