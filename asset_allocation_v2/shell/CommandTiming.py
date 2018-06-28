#coding=utf8


import getopt
import string
import json
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import LabelAsset
import os
import time
import re
import Const
import DFUtil
from TimingGFTD import TimingGFTD
from TimingHmm import TimingHmm

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from tabulate import tabulate
from db import *
from util.xdebug import dd

import traceback, code

@click.group(invoke_without_command=True)
@click.option('--id', 'optid', help='fund pool id to update')
@click.option('--online/--no-online', 'optonline', default=False, help='include online instance')
@click.pass_context
def timing(ctx, optid, optonline):
    '''timing group
    '''
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        ctx.invoke(signal, optid=optid, optonline=optonline)
        ctx.invoke(nav, optid=optid)
    else:
        # click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        pass

@timing.command()
@click.option('--id', 'optid', help='fund pool id to update')
@click.option('--list/--no-list', 'optlist', default=False, help='list pool to update')
@click.option('--online/--no-online', 'optonline', default=False, help='include online instance')
@click.pass_context
def signal(ctx, optid, optlist, optonline):
    '''calc timing signal  for timing instance
    '''
    if optid is not None:
        timings = [s.strip() for s in optid.split(',')]
    else:
        timings = None

    xtypes = None
    if optonline == False:
        xtypes = [1]

    df_timing = asset_tc_timing.load(timings, xtypes)


    if optlist:

        df_timing['tc_name'] = df_timing['tc_name'].map(lambda e: e.decode('utf-8'))
        return 0
    
    with click.progressbar(length=len(df_timing), label='update signal'.ljust(30)) as bar:
        for _, timing in df_timing.iterrows():
            bar.update(1)
            if timing['tc_method'] == 1:
                signal_update_gftd(timing)
            elif timing['tc_method'] == 3:
                signal_update_hmm(timing)



def signal_update_hmm(timing):
    '''calc timing signal for singe timing instance
    '''
    #
    # 加载OHLC数据
    #
    timing_id = timing['globalid']
    yesterday = (datetime.now() - timedelta(days=1));

    sdate = timing['tc_begin_date'].strftime("%Y-%m-%d")
    edate = yesterday.strftime("%Y-%m-%d")

    df_nav = base_ra_index_nav.load_ohlcav(
        timing['tc_index_id'], end_date=edate, mask=[0, 1, 2])

    tdates = base_trade_dates.load_index(begin_date = df_nav.index[0])
    df_nav = df_nav.loc[tdates].dropna()

    df_nav.rename(columns={'ra_open':'tc_open', 'ra_high':'tc_high', 'ra_low':'tc_low', 'ra_close':'tc_close', 'ra_volume':'tc_volume', 'ra_amount':'tc_amount'}, inplace=True)
    df_nav.index.name='tc_date'

    # risk_mgr = RiskManagement.RiskManagement()
    trade_dates = base_trade_dates.load_trade_dates()
    df_new = TimingHmm(ori_data = df_nav, timing = timing, trade_dates = trade_dates, start_date = df_nav.index[252 * 5] if df_nav.index[252 * 5] > datetime(2012,5,1) else '2012-05-01').timing()
    df_new = df_new[['tc_signal']]

    df_new['tc_timing_id'] = timing_id
    df_new = df_new.reset_index().set_index(['tc_timing_id', 'tc_date'])

    #
    # 保存择时结果到数据库
    #
    db = database.connection('asset')

    # 更新tc_timing_signal
    df_new = df_new[['tc_signal']]
    t3 = Table('tc_timing_signal', MetaData(bind=db), autoload=True)
    columns3 = [
        t3.c.tc_timing_id,
        t3.c.tc_date,
        t3.c.tc_signal,
    ]
    s = select(columns3, (t3.c.tc_timing_id == timing_id))
    df_old = pd.read_sql(s, db, index_col=['tc_timing_id', 'tc_date'], parse_dates=['tc_date'])

    # 更新数据库
    database.batch(db, t3, df_new, df_old, timestamp=False)
    return 0


def signal_update_gftd(timing):
    '''calc timing signal for singe timing instance
    '''
    #
    # 加载OHLC数据
    #
    timing_id = timing['globalid']
    yesterday = (datetime.now() - timedelta(days=1));

    sdate = timing['tc_begin_date'].strftime("%Y-%m-%d")
    edate = yesterday.strftime("%Y-%m-%d")

    #tdates = base_trade_dates.load_index(sdate, edate)
    tdates = base_trade_dates.load_origin_index_trade_date(timing['tc_index_id'], sdate)
    #
    # 解析模型参数
    #
    argv = {k: int(v) for (k,v) in [x.split('=') for x in timing['tc_argv'].split(',')]}
    n1, n2, n3, n4 = argv['n1'], argv['n2'], argv['n3'], argv['n4']

    df_nav = load_index_ohlc(
        timing['tc_index_id'], reindex=tdates, begin_date=sdate, end_date=None, mask=[0, 2])
    df_nav = df_nav.sort_index()


    # risk_mgr = RiskManagement.RiskManagement()
    df_new = TimingGFTD(n1=n1,n2=n2,n3=n3,n4=n4).timing(df_nav)
    df_new['tc_timing_id'] = timing_id
    df_new = df_new.reset_index().set_index(['tc_timing_id', 'tc_date'])
    # print df_new[df_new['tc_stop'].isnull()].head()

    num_signal = df_new['tc_signal'].rolling(2, 1).apply(lambda x: 1 if x[-1] != x[0] else 0).sum()

    formaters = ['tc_close', 'tc_open', 'tc_high', 'tc_low', 'tc_recording_high', 'tc_recording_low', 'tc_stop_high', 'tc_stop_low']

    if not df_new.empty:
        df_new = database.number_format(df_new, columns=formaters, precision=4)

    #
    # 保存择时结果到数据库
    #
    db = database.connection('asset')
    t2 = Table('tc_timing_scratch', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.tc_timing_id,
        t2.c.tc_date,
        t2.c.tc_open,
        t2.c.tc_high,
        t2.c.tc_low,
        t2.c.tc_close,
        t2.c.tc_ud,
        # t2.c.tc_ud_flip,
        t2.c.tc_ud_acc,
        t2.c.tc_buy_start,
        # t2.c.tc_buy_kstick,
        t2.c.tc_buy_count,
        t2.c.tc_buy_signal,
        t2.c.tc_sell_start,
        # t2.c.tc_sell_kstick,
        t2.c.tc_sell_count,
        t2.c.tc_sell_signal,
        t2.c.tc_action,
        t2.c.tc_recording_high,
        t2.c.tc_recording_low,
        t2.c.tc_signal,
        t2.c.tc_stop_high,
        t2.c.tc_stop_low,
    ]
    s = select(columns2, (t2.c.tc_timing_id == timing_id))
    df_old = pd.read_sql(s, db, index_col=['tc_timing_id', 'tc_date'], parse_dates=['tc_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, columns=formaters, precision=4)

    # 更新数据库
    df_old = df_old[df_new.columns]
    database.batch(db, t2, df_new, df_old, timestamp=False)
    # print "total signal: %d, %.2f/year" % (num_signal, num_signal * 250/len(df_new))

    # 更新tc_timing_signal
    df_new = df_new[['tc_signal']]
    t3 = Table('tc_timing_signal', MetaData(bind=db), autoload=True)
    columns3 = [
        t3.c.tc_timing_id,
        t3.c.tc_date,
        t3.c.tc_signal,
    ]
    s = select(columns3, (t3.c.tc_timing_id == timing_id))
    df_old = pd.read_sql(s, db, index_col=['tc_timing_id', 'tc_date'], parse_dates=['tc_date'])

    # 更新数据库
    database.batch(db, t3, df_new, df_old, timestamp=False)

@timing.command()
@click.option('--id', 'optid', help='ids of fund pool to update')
@click.option('--list/--no-list', 'optlist', default=False, help='list pool to update')
@click.pass_context
def nav(ctx, optid, optlist):
    ''' calc pool nav and inc
    '''
    if optid is not None:
        timings = [s.strip() for s in optid.split(',')]
    else:
        timings = None

    df_timing = asset_tc_timing.load(timings)

    if optlist:

        df_timing['tc_name'] = df_timing['tc_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_timing, headers='keys', tablefmt='psql'))
        return 0
    
    with click.progressbar(length=len(df_timing), label='update nav'.ljust(30)) as bar:
        for _, timing in df_timing.iterrows():
            bar.update(1)
            nav_update(timing)

def nav_update(timing):
    timing_id = timing['globalid']
    # 加载择时信号
    # df_position = database.asset_tc_timing_scratch_load_signal(timing_id)
    df_position = asset_tc_timing_signal.load(timing_id)
    # 构建仓位
    df_position.loc[df_position[timing_id] < 1, timing_id] = 0

    
    # 加载基金收益率
    min_date = df_position.index.min()
    #max_date = df_position.index.max()
    max_date = (datetime.now() - timedelta(days=1)) # yesterday


    df_nav = base_ra_index_nav.load_series(
        timing['tc_index_id'], begin_date=min_date, end_date=max_date, mask=0)
    df_inc = df_nav.pct_change().fillna(0.0).to_frame(timing_id)

    # 计算复合资产净值
    df_nav_portfolio = DFUtil.portfolio_nav(df_inc, df_position, result_col='portfolio')

    df_result = df_nav_portfolio[['portfolio']].rename(columns={'portfolio':'tc_nav'}).copy()
    df_result.index.name = 'tc_date'
    df_result['tc_inc'] = df_result['tc_nav'].pct_change().fillna(0.0)
    df_result['tc_timing_id'] = timing['globalid']
    df_result = df_result.reset_index().set_index(['tc_timing_id', 'tc_date'])
    
    df_new = database.number_format(df_result, columns=['tc_nav', 'tc_inc'], precision=6)

    # 加载旧数据
    db = database.connection('asset')
    t2 = Table('tc_timing_nav', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.tc_timing_id,
        t2.c.tc_date,
        t2.c.tc_nav,
        t2.c.tc_inc,
    ]
    stmt_select = select(columns2, (t2.c.tc_timing_id == timing['globalid']))
    df_old = pd.read_sql(stmt_select, db, index_col=['tc_timing_id', 'tc_date'], parse_dates=['tc_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, columns=['tc_nav', 'tc_inc'], precision=6)

    # 更新数据库
    database.batch(db, t2, df_new, df_old, timestamp=False)
    
    #print df_result.head()

@timing.command()
@click.option('--id', 'optid', help='id of timing to generate coverage')
@click.option('--list/--no-list', 'optlist', default=False, help='list pool to update')
@click.option('--n1', 'optn1', default='4', help='n1 arguments')
@click.option('--n2', 'optn2', default='4', help='n2 arguments')
@click.option('--n3', 'optn3', default='4', help='n3 arguments')
@click.option('--n4', 'optn4', default='4', help='n4 arguments')
@click.pass_context
def coverage(ctx, optid, optlist, optn1, optn2, optn3, optn4):
    ''' calc pool coverage and inc
    '''
    if optid is not None:
        timings = [s.strip() for s in optid.split(',')]
    else:
        timings = None

    n1s = [int(s.strip()) for s in optn1.split(',')]
    n2s = [int(s.strip()) for s in optn2.split(',')]
    n3s = [int(s.strip()) for s in optn3.split(',')]
    n4s = [int(s.strip()) for s in optn4.split(',')]

    df_timing = asset_tc_timing.load(timings)

    if optlist:

        df_timing['tc_name'] = df_timing['tc_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_timing, headers='keys', tablefmt='psql'))
        return 0

    with click.progressbar(
            df_timing.iterrows(), length=len(df_timing.index),
            label=('update %-13s' % 'coverage').ljust(30),
            item_show_func=lambda x:  str(x[1]['globalid']) if x else None) as bar:
        for _, timing in bar:
            coverage_update(timing, n1s, n2s, n3s, n4s)

def coverage_update(timing, n1s, n2s, n3s, n4s):
    timing_id = timing['globalid']

    min_id = timing_id
    max_id = int(timing_id / 100) * 100 + 99

    gid = start_id = asset_tc_timing.max_id_between(min_id, max_id)
    
    argv = 'n1=%d,n2=%d,n3=%d,n4=%d'
    name = timing['tc_name']
    xtype = 1
    method = 1

    data = []
    for n1 in n1s:
        for n2 in n2s:
            for n3 in n3s:
                for n4 in n4s:
                    gid += 1
                    if gid > max_id:
                        click.echo(click.style("run out of globalid", fg='red'))
                        sys.exit(-1)
                    name = '%s%d%d%d%d' % (timing['tc_name'], n1, n2, n3, n4)
                    argv = 'n1=%d,n2=%d,n3=%d,n4=%d' % (n1, n2, n3, n4)
                    data.append((gid, name, xtype, method, timing['tc_index_id'], timing['tc_begin_date'], argv))
    df = pd.DataFrame(data, columns=['globalid', 'tc_name', 'tc_type', 'tc_method', 'tc_index_id', 'tc_begin_date', 'tc_argv'])
    df = df.set_index(['globalid'])
    
    # 加载旧数据
    db = database.connection('asset')
    t2 = Table('tc_timing', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.globalid,
        t2.c.tc_name,
        t2.c.tc_type,
        t2.c.tc_method,
        t2.c.tc_index_id,
        t2.c.tc_begin_date,
        t2.c.tc_argv,
    ]
    stmt_select = select(columns2, (t2.c.globalid.between(start_id + 1, max_id)))
    df_old = pd.read_sql(stmt_select, db, index_col=['globalid'])

    # 更新数据库
    database.batch(db, t2, df, df_old, timestamp=False)
   


def load_index_ohlc(asset_id, reindex, begin_date, end_date, mask):

    if asset_id.isdigit():
        xtype = int(asset_id) // 10000000
    else:
        xtype = re.sub(r'([\d]+)','',asset_id).strip()

    if xtype == 'ERI':
        df_nav = base_exchange_rate_index_nav.load_ohlc(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date, mask = mask)

        df_nav.rename(columns={'eri_open':'tc_open', 'eri_high':'tc_high', 'eri_low':'tc_low', 'eri_close':'tc_close'}, inplace=True)
        df_nav.index.name='tc_date'

    elif xtype == 12:
        df_nav = base_ra_index_nav.load_ohlc(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date, mask = mask)

        df_nav.rename(columns={'ra_open':'tc_open', 'ra_high':'tc_high', 'ra_low':'tc_low', 'ra_close':'tc_close'}, inplace=True)
        df_nav.index.name='tc_date'

    #print df_nav.head()
    return df_nav

