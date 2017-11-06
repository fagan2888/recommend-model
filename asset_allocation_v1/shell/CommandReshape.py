#coding=utf8


import string
import json
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import os
import time
import logging
import Const
import DFUtil
import util_numpy as npu

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from tabulate import tabulate
from db import *
from Reshape import Reshape


import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)  
@click.option('--id', 'optid', help=u'reshape id')
@click.option('--online/--no-online', 'optonline', default=False, help=u'include online instance')
@click.pass_context
def reshape(ctx, optid, optonline):
    '''reshape group
    '''
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        ctx.invoke(pos, optid=optid, optonline=optonline)
        ctx.invoke(nav, optid=optid)
    else:
        # click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        pass

@reshape.command(name='import')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace pool if exists')
@click.argument('csv', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=False), required=True)
@click.pass_context
def import_command(ctx, csv, opttype, optreplace):
    '''
    import reshape from csv file
    '''

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    rs_reshape = Table('rs_reshape', metadata, autoload=True)
    rs_reshape_pos = Table('rs_reshape_pos', metadata, autoload=True)
    rs_reshape_nav = Table('rs_reshape_nav', metadata, autoload=True)


    now = datetime.now()

    df_csv = pd.read_csv(csv, parse_dates=['date'])
    renames = dict(
        {'date':'rs_date'}.items() + DFUtil.categories_types(as_int=True).items()
    )
    df_csv = df_csv.rename(columns=renames)
    df_csv.set_index(['rs_date'], inplace=True)

    for column in df_csv.columns:
        if column in [21, 22, 23, 31]:
            continue
        
        optid = "4%s2%d01%d" % (opttype, int(column) // 10, column)

        #
        # 处理替换
        #
        if optreplace:
            # rs_reshape.delete(rs_reshape.c.globalid == optid).execute()
            rs_reshape_pos.delete(rs_reshape_pos.c.rs_reshape_id == optid).execute()
            rs_reshape_nav.delete(rs_reshape_nav.c.rs_reshape_id == optid).execute()
            df = df_csv[[column]].copy()
        
        df['rs_reshape_id'] = optid
        df = df.reset_index().set_index(['rs_reshape_id', 'rs_date'])

        # 四舍五入到万分位
        df = df.round(4)
        # 过滤掉过小的份额
        df[df.abs() < 0.0009999] = 0
        # 过滤掉相同
        df = DFUtil.filter_same_with_last(df)

        df = df.rename(columns={column: 'rs_ratio'})
        if not df.empty:
            database.number_format(df, columns=['rs_ratio'], precision=4)

        df['updated_at'] = df['created_at'] = now

        df.to_sql(rs_reshape_pos.name, db, index=True, if_exists='append', chunksize=500)

        if len(df.index) > 1:
            logger.info("insert %s (%5d) : %s " % (rs_reshape_pos.name, len(df.index), df.index[0]))

        click.echo(click.style("import complement! instance id [%s]" % (optid), fg='green'))


@reshape.command()
@click.option('--id', 'optid', help=u'ids of fund pool to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.pass_context
def nav(ctx, optid, optlist):
    ''' calc reshape nav and inc
    '''
    if optid is not None:
        ids = [s.strip() for s in optid.split(',')]
    else:
        ids = None

    df_reshape = asset_rs_reshape.load(ids)

    if optlist:

        df_reshape['rs_name'] = df_reshape['rs_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_reshape, headers='keys', tablefmt='psql')
        return 0
    
    with click.progressbar(length=len(df_reshape), label='update nav'.ljust(30)) as bar:
        for _, reshape in df_reshape.iterrows():
            bar.update(1)
            nav_update(reshape)

def nav_update(reshape):
    reshape_id = reshape['globalid']
    # 加载择时信号
    df_position = asset_rs_reshape_pos.load([reshape_id])
    if df_position.empty:
        return 

    # 加载基金收益率
    min_date = df_position.index.min()
    #max_date = df_position.index.max()
    max_date = (datetime.now() - timedelta(days=1)) # yesterday


    sr_nav = database.load_nav_series(
        reshape['rs_asset_id'], begin_date=min_date, end_date=max_date);
    df_inc = sr_nav.pct_change().fillna(0.0).to_frame(reshape_id)

    # 计算复合资产净值
    df_nav_portfolio = DFUtil.portfolio_nav(df_inc, df_position, result_col='portfolio')

    df_result = df_nav_portfolio[['portfolio']].rename(columns={'portfolio':'rs_nav'}).copy()
    df_result.index.name = 'rs_date'
    df_result['rs_inc'] = df_result['rs_nav'].pct_change().fillna(0.0)
    df_result['rs_reshape_id'] = reshape['globalid']
    df_result = df_result.reset_index().set_index(['rs_reshape_id', 'rs_date'])
    
    df_new = database.number_format(df_result, columns=['rs_nav', 'rs_inc'], precision=6)

    # 加载旧数据
    db = database.connection('asset')
    t2 = Table('rs_reshape_nav', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.rs_reshape_id,
        t2.c.rs_date,
        t2.c.rs_nav,
        t2.c.rs_inc,
    ]
    stmt_select = select(columns2, (t2.c.rs_reshape_id == reshape['globalid']))
    df_old = pd.read_sql(stmt_select, db, index_col=['rs_reshape_id', 'rs_date'], parse_dates=['rs_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, columns=['rs_nav', 'rs_inc'], precision=6)

    # 更新数据库
    database.batch(db, t2, df_new, df_old, timestamp=True)

@reshape.command()
@click.option('--id', 'optid', help=u'ids of fund pool to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.option('--online/--no-online', 'optonline', default=False, help=u'include online instance')
@click.pass_context
def pos(ctx, optid, optlist, optonline):
    ''' calc reshape pos and inc
    '''
    if optid is not None:
        ids = [s.strip() for s in optid.split(',')]
    else:
        ids = None

    xtypes = None
    if optonline == False:
        xtypes = [1]

    df_reshape = asset_rs_reshape.load(ids, xtypes)

    if optlist:

        df_reshape['rs_name'] = df_reshape['rs_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_reshape, headers='keys', tablefmt='psql')
        return 0
    
    with click.progressbar(length=len(df_reshape), label='update pos'.ljust(30)) as bar:
        for _, reshape in df_reshape.iterrows():
            bar.update(1)
            pos_update(reshape)

def pos_update(reshape):
    reshape_id = reshape['globalid']
    
    # 加载择时信号
    sr_timing = asset_tc_timing_signal.load_series(reshape['rs_timing_id'])
    # print sr_timing.head()
    
    # 加载资产收益率
    # min_date = df_position.index.min()
    # max_date = (datetime.now() - timedelta(days=1)) # yesterday

    if reshape['rs_start_date'] != '0000-00-00':
        sdate = reshape['rs_start_date']
    else:
        sdate = None

    tdates = base_trade_dates.load_index(sdate)

    sr_nav = database.load_nav_series(reshape['rs_asset_id'], reindex=tdates, begin_date=sdate)
    
    # df_inc = df_nav.pct_change().fillna(0.0).to_frame(reshape_id)
    df = pd.DataFrame({'nav': sr_nav, 'timing': sr_timing})

    df_result = Reshape().reshape(df)
    df_result.drop(['nav', 'timing'], axis=1, inplace=True)
    
    # df_result = df_nav_portfolio[['portfolio']].rename(columns={'portfolio':'rs_nav'}).copy()
    df_result.index.name = 'rs_date'
    df_result['rs_reshape_id'] = reshape_id
    df_result['rs_action'] = 0
    df_new = df_result.reset_index().set_index(['rs_reshape_id', 'rs_date'])

    fmt_columns = ['rs_r20','rs_return', 'rs_return_mean', 'rs_return_std', 'rs_risk', 'rs_risk_mean', 'rs_risk_std']
    df_new = database.number_format(df_new, fmt_columns, 6, rs_ratio=4)

    #print df_new.head()
    

    # 加载旧数据
    db = database.connection('asset')
    t2 = Table('rs_reshape_pos', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.rs_reshape_id,
        t2.c.rs_date,
        t2.c.rs_r20,
        t2.c.rs_return,
        t2.c.rs_risk,
        t2.c.rs_return_mean,
        t2.c.rs_return_std,
        t2.c.rs_risk,
        t2.c.rs_risk_mean,
        t2.c.rs_risk_std,
        t2.c.rs_ratio,
        t2.c.rs_action,
    ]
    stmt_select = select(columns2, (t2.c.rs_reshape_id == reshape_id))
    df_old = pd.read_sql(stmt_select, db, index_col=['rs_reshape_id', 'rs_date'], parse_dates=['rs_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, fmt_columns, 6, rs_ratio=4)

    # 更新数据库
    database.batch(db, t2, df_new, df_old, timestamp=True)
    
