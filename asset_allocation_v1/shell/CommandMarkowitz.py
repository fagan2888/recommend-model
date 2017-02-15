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
import Const
import DFUtil
import DBData
import util_numpy as npu
import Portfolio as PF

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from tabulate import tabulate
from db import database, asset_mz_markowitz, asset_mz_markowitz_asset, asset_mz_markowitz_nav, asset_mz_markowitz_pos, asset_mz_markowitz_sharpe
from db import asset_ra_pool, asset_ra_pool_nav, asset_rs_reshape, asset_rs_reshape_nav
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav
from util import xdict

import traceback, code

logger = logging.getLogger(__name__)

@click.group()  
@click.pass_context
def markowitz(ctx):
    '''markowitz group
    '''
    pass



@markowitz.command(name='import')
@click.option('--id', 'optid', type=int, help=u'specify markowitz id')
@click.option('--name', 'optname', help=u'specify markowitz name')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace pool if exists')
@click.argument('csv', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=False), required=True)
@click.pass_context
def import_command(ctx, csv, optid, optname, opttype, optreplace):
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
        df_existed = asset_mz_markowitz.load([str(optid)])
        if not df_existed.empty:
            s = 'markowitz instance [%d] existed' % optid
            if optreplace:
                click.echo(click.style("%s, will replace!" % s, fg="yellow"))
            else:
                click.echo(click.style("%s, import aborted!" % s, fg="red"))
            return -1;
    else:
        #
        # 自动生成id
        #
        today = datetime.now()
        prefix = '50' + today.strftime("%m%d");
        if opttype == '9':
            between_min, between_max = ('%s90' % (prefix), '%s99' % (prefix))
        else:
            between_min, between_max = ('%s00' % (prefix), '%s89' % (prefix))

        max_id = asset_mz_markowitz.max_id_between(between_min, between_max)
        if max_id is None:
            optid = between_min
        else:
            if max_id >= between_max:
                s = "run out of instance id [%d]" % max_id
                click.echo(click.style("%s, import aborted!" % s, fg="red"))
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
    mz_markowitz = Table('mz_markowitz', metadata, autoload=True)
    mz_markowitz_pos = Table('mz_markowitz_pos', metadata, autoload=True)
    mz_markowitz_nav = Table('mz_markowitz_nav', metadata, autoload=True)

    #
    # 处理替换
    #
    if optreplace:
        mz_markowitz.delete(mz_markowitz.c.globalid == optid).execute()
        mz_markowitz_pos.delete(mz_markowitz_pos.c.mz_markowitz_id == optid).execute()
        mz_markowitz_nav.delete(mz_markowitz_nav.c.mz_markowitz_id == optid).execute()

    now = datetime.now()
    #
    # 导入数据
    #
    row = {
        'globalid': optid, 'mz_type':opttype, 'mz_name': optname,
        'mz_pool': '', 'mz_reshape': '', 'created_at': func.now(), 'updated_at': func.now()
    }
    mz_markowitz.insert(row).execute()

    df = pd.read_csv(csv, parse_dates=['date'])
    df['risk'] = (df['risk'] * 10).astype(int)
    renames = dict(
        {'date':'mz_date', 'risk':'mz_alloc_id'}.items() + DFUtil.categories_types(as_int=True).items()
    )
    df = df.rename(columns=renames)
    df['mz_markowitz_id'] = optid

    df.set_index(['mz_markowitz_id', 'mz_alloc_id', 'mz_date'], inplace=True)

    # 四舍五入到万分位
    df = df.round(4)
    # 过滤掉过小的份额
    df[df.abs() < 0.0009999] = 0
    # 补足缺失
    df = df.apply(npu.np_pad_to, raw=True, axis=1)
    # 过滤掉相同
    df = df.groupby(level=(0,1), group_keys=False).apply(DFUtil.filter_same_with_last)

    df.columns.name='mz_asset'
    df_tosave = df.stack().to_frame('mz_ratio')
    df_tosave = df_tosave.loc[df_tosave['mz_ratio'] > 0, ['mz_ratio']]
    if not df_tosave.empty:
        database.number_format(df_tosave, columns=['mz_ratio'], precision=4)
    
    df_tosave['updated_at'] = df_tosave['created_at'] = now

    df_tosave.to_sql(mz_markowitz_pos.name, db, index=True, if_exists='append', chunksize=500)

    if len(df_tosave.index) > 1:
        logger.info("insert %s (%5d) : %s " % (mz_markowitz_pos.name, len(df_tosave.index), df_tosave.index[0]))

    click.echo(click.style("import complement! instance id [%s]" % (optid), fg='green'))
    
    return 0

@markowitz.command()
@click.option('--id', 'optid', type=int, help=u'specify markowitz id')
@click.option('--name', 'optname', default=u'马克维茨', help=u'specify markowitz name')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace pool if exists')
@click.option('--start-date', 'startdate', default='2012-07-27', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.option('--lookback', type=int, default=26, help=u'howmany weeks to lookback')
@click.option('--adjust-period', type=int, default=1, help=u'adjust every how many weeks')
@click.argument('assets', nargs=-1)
@click.pass_context
def allocate(ctx, optid, optname, opttype, optreplace, startdate, enddate, lookback, adjust_period, assets):
    '''calc high low model markowitz
    '''

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")

    #
    # 处理id参数
    #
    if optid is not None:
        #
        # 检查id是否存在
        #
        df_existed = asset_mz_markowitz.load([str(optid)])
        if not df_existed.empty:
            s = 'markowitz instance [%d] existed' % optid
            if optreplace:
                click.echo(click.style("%s, will replace!" % s, fg="yellow"))
            else:
                click.echo(click.style("%s, import aborted!" % s, fg="red"))
            return -1;
    else:
        #
        # 自动生成id
        #
        today = datetime.now()
        prefix = '50' + today.strftime("%m%d");
        if opttype == '9':
            between_min, between_max = ('%s90' % (prefix), '%s99' % (prefix))
        else:
            between_min, between_max = ('%s00' % (prefix), '%s89' % (prefix))

        max_id = asset_mz_markowitz.max_id_between(between_min, between_max)
        if max_id is None:
            optid = between_min
        else:
            if max_id >= between_max:
                s = "run out of instance id [%d]" % max_id
                click.echo(click.style("%s, import aborted!" % s, fg="red"))
                return -1

            if optreplace:
                optid = max_id
            else:
                optid = max_id + 1;
    #
    # 处理assets参数
    #
    if assets:
        assets = {k: v for k,v in [parse_asset(a) for a in assets]}
    else:
        assets = {
            41210111:  {'sumlimit': 0, 'uplimit': 1.0, 'downlimit': 0.0},
            41210112:  {'sumlimit': 0, 'uplimit': 1.0, 'downlimit': 0.0},
            41210113:  {'sumlimit': 0, 'uplimit': 1.0, 'downlimit': 0.0},
            41210115:  {'sumlimit': 0, 'uplimit': 1.0, 'downlimit': 0.0},
            41210116:  {'sumlimit': 0, 'uplimit': 1.0, 'downlimit': 0.0},
            41210117:  {'sumlimit': 0, 'uplimit': 1.0, 'downlimit': 0.0},
            120000013: {'sumlimit': 1, 'uplimit': 0.3, 'downlimit': 0.0},
            120000014: {'sumlimit': 1, 'uplimit': 0.3, 'downlimit': 0.0},
            120000015: {'sumlimit': 1, 'uplimit': 0.3, 'downlimit': 0.0},
        }

    df = markowitz_days(startdate, enddate, assets,
        label=optname, lookback=lookback, adjust_period=adjust_period)

    df_sharpe = df[['return', 'risk', 'sharpe']].copy()
    df.drop(['return', 'risk', 'sharpe'], axis=1, inplace=True)
    
    # print df.head()
    # print df_sharpe.head()
    
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    mz_markowitz        = Table('mz_markowitz', metadata, autoload=True)
    mz_markowitz_asset  = Table('mz_markowitz_asset', metadata, autoload=True)
    mz_markowitz_pos    = Table('mz_markowitz_pos', metadata, autoload=True)
    mz_markowitz_nav    = Table('mz_markowitz_nav', metadata, autoload=True)
    mz_markowitz_sharpe = Table('mz_markowitz_sharpe', metadata, autoload=True)

    #
    # 处理替换
    #
    if optreplace:
        mz_markowitz.delete(mz_markowitz.c.globalid == optid).execute()
        mz_markowitz_asset.delete(mz_markowitz_asset.c.mz_markowitz_id == optid).execute()
        mz_markowitz_pos.delete(mz_markowitz_pos.c.mz_markowitz_id == optid).execute()
        mz_markowitz_nav.delete(mz_markowitz_nav.c.mz_markowitz_id == optid).execute()
        mz_markowitz_sharpe.delete(mz_markowitz_sharpe.c.mz_markowitz_id == optid).execute()

    now = datetime.now()
    # 导入数据: markowitz
    row = {
        'globalid': optid, 'mz_type':opttype, 'mz_name': optname,
        'created_at': func.now(), 'updated_at': func.now()
    }
    mz_markowitz.insert(row).execute()
    # 导入数据: markowitz_asset
    assets = {k: merge_asset_name_and_type(k, v) for (k, v) in assets.iteritems()}
    df_asset = pd.DataFrame(assets).T
    df_asset.index.name = 'mz_asset_id'
    df_asset['mz_markowitz_id'] = optid
    df_asset.rename(inplace=True, columns={
        'uplimit':'mz_upper_limit', 'downlimit':'mz_lower_limit', 'sumlimit':'mz_sum1_limit'})
    df_asset = df_asset.reset_index().set_index(['mz_markowitz_id', 'mz_asset_id'])
    print df_asset
    asset_mz_markowitz_asset.save(optid, df_asset)
    # 导入数据: markowitz_pos
    df = df.round(4)             # 四舍五入到万分位
    df[df.abs() < 0.0009999] = 0 # 过滤掉过小的份额
    # print df.head()
    df = df.apply(npu.np_pad_to, raw=True, axis=1) # 补足缺失
    df = DFUtil.filter_same_with_last(df)          # 过滤掉相同
    # index
    df['mz_markowitz_id'] = optid
    df.index.name = 'mz_date'
    df = df.reset_index().set_index(['mz_markowitz_id', 'mz_date'])
    # unstack
    df.columns.name='mz_asset_id'
    df_tosave = df.stack().to_frame('mz_ratio')
    df_tosave = df_tosave.loc[df_tosave['mz_ratio'] > 0, ['mz_ratio']]
    # save
    asset_mz_markowitz_pos.save(optid, df_tosave)
    # 导入数据: markowitz_sharpe
    df_sharpe.index.name = 'mz_date'
    df_sharpe['mz_markowitz_id'] = optid
    df_sharpe.rename(inplace=True, columns={
        'return': 'mz_return', 'risk':'mz_risk', 'sharpe':'mz_sharpe'})
    # print df_sharpe
    df_sharpe = df_sharpe.reset_index().set_index(['mz_markowitz_id', 'mz_date'])
    asset_mz_markowitz_sharpe.save(optid, df_sharpe)
    
    click.echo(click.style("import complement! instance id [%s]" % (optid), fg='green'))

def parse_asset(asset):
    segments = [s.strip() for s in asset.strip().split(':')]

    if len(segments) == 1:
        result = (int(segments[0]), {
            'uplimit': 1.0, 'downlimit': 0.0, 'sumlimit': 0})
    elif len(segments) == 2:
        result = (int(segments[0]), {
            'uplimit': float(segments[1]), 'downlimit': 0.0, 'sumlimit': 0})
    elif len(segments) == 3:
        result = (int(segments[0]), {
            'uplimit': float(segments[1]), 'downlimit': float(segments[2]), 'sumlimit': 0})
    else:
        if len(segments) >= 4:
            sumlimit = 1 if segments[3] == '1' else 0
            result = (int(segments[0]), {
                'uplimit': float(segments[1]), 'downlimit': float(segments[2]), 'sumlimit': sumlimit})
        else:
            result = (None, {'uplimit': 1.0, 'downlimit': 0.0, 'sumlimit': 0})

    return result

def merge_asset_name_and_type(asset_id, asset_data):
    (name, category) = load_asset_name_and_type(asset_id)
    return xdict.merge(asset_data, {
        'mz_asset_name': name, 'mz_asset_type': category})

            

def markowitz_days(start_date, end_date, assets, label, lookback, adjust_period):
    '''perform markowitz asset for days
    '''
    # 加载时间轴数据
    index = DBData.trade_date_index(start_date, end_date=end_date)

    # 根据调整间隔抽取调仓点
    if adjust_period:
        adjust_index = index[::adjust_period]
        if index.max() not in adjust_index:
            adjust_index = adjust_index.insert(len(adjust_index), index.max())
    else:
        adjust_index = index

    #
    # 马科维兹资产配置
    #
    s = 'perform %s' % label
    data = {}
    with click.progressbar(length=len(adjust_index), label=s) as bar:
        for day in adjust_index:
            bar.update(1)
            logger.debug("%s : %s", s, day.strftime("%Y-%m-%d"))
            # 高风险资产配置
            data[day] = markowitz_day(day, lookback, assets)

    return pd.DataFrame(data).T

def markowitz_day(day, lookback, assets):
    '''perform markowitz for single day
    '''
    
    # 加载时间轴数据
    index = DBData.trade_date_lookback_index(end_date=day, lookback=lookback)
    begin_date = index.min().strftime("%Y-%m-%d")
    end_date = index.max().strftime("%Y-%m-%d")

    #
    # 加载数据
    #
    data = {}
    for asset in assets:
        data[asset] = load_nav_series(asset, begin_date, end_date)
    df_nav = pd.DataFrame(data).fillna(method='pad')
    df_inc  = df_nav.pct_change().fillna(0.0)
    #
    # 根据时间轴进行重采样
    #
    df_inc = df_inc.reindex(index, fill_value=0.0)

    return markowitz_r(df_inc, assets)

def markowitz_r(df_inc, limits):
    '''perform markowitz
    '''
    bound = []
    for asset in df_inc.columns:
        bound.append(limits[asset])
    
    risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)

    sr_result = pd.concat([
        pd.Series(ws, index=df_inc.columns),
        pd.Series((sharpe, risk, returns), index=['sharpe','risk', 'return'])
    ])

    return sr_result

def load_nav_series(asset_id, begin_date, end_date):
    xtype = asset_id / 10000000

    if xtype == 1:
        #
        # 基金池资产
        #
        asset_id %= 10000000
        (pool_id, category) = (asset_id / 100, asset_id % 100)
        ttype = pool_id / 10000
        sr = asset_ra_pool_nav.load_series(
            pool_id, category, ttype, begin_date=begin_date, end_date=end_date)
    elif xtype == 3:
        #
        # 基金池资产
        #
        sr = base_ra_fund_nav.load_series(
            asset_id, begin_date=begin_date, end_date=end_date)
    elif xtype == 4:
        #
        # 修型资产
        #
        sr = asset_rs_reshape_nav.load_series(asset_id, begin_date, end_date)
    elif xtype == 12:
        #
        # 指数资产
        #
        sr = base_ra_index_nav.load_series(
            asset_id, begin_date=begin_date, end_date=end_date)
    else:
        sr = pd.Series()

    return sr

def load_asset_name_and_type(asset_id):
    (name, category) = ('', 0)
    xtype = asset_id / 10000000

    if xtype == 1:
        #
        # 基金池资产
        #
        asset_id %= 10000000
        (pool_id, category) = (asset_id / 100, asset_id % 100)
        ttype = pool_id / 10000
        name = asset_ra_pool.load_asset_name(pool_id, category, ttype)
    elif xtype == 3:
        #
        # 基金池资产
        #
        category = 1
        fund = base_ra_fund.find(asset_id)
        name = "%s(%s)" % (fund['ra_name'], fund['ra_code'])
        
    elif xtype == 4:
        #
        # 修型资产
        #
        asset = asset_rs_reshape.find(asset_id)
        (name, category) = (asset['rs_name'], asset['rs_asset'])
    elif xtype == 12:
        #
        # 指数资产
        #
        asset = base_ra_index.find(asset_id)
        name = asset['ra_name']
        if '标普' in name:
            category = 41
        elif '黄金' in name:
            category = 42
        elif '恒生' in name:
            category = 43
    else:
         (name, category) = ('', 0)

    return (name, category)


@markowitz.command()
@click.option('--id', 'optid', help=u'ids of markowitz to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.pass_context
def nav(ctx, optid, optlist):
    ''' calc pool nav and inc
    '''
    if optid is not None:
        markowitzs = [s.strip() for s in optid.split(',')]
    else:
        markowitzs = None

    df_markowitz = asset_mz_markowitz.load(markowitzs)

    if optlist:

        df_markowitz['mz_name'] = df_markowitz['mz_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_markowitz, headers='keys', tablefmt='psql')
        return 0
    
    with click.progressbar(length=len(df_markowitz), label='update nav') as bar:
        for _, markowitz in df_markowitz.iterrows():
            bar.update(1)
            nav_update(markowitz)

def nav_update(markowitz):
    markowitz_id = markowitz['globalid']
    # 加载仓位信息
    df_pos = asset_mz_markowitz_pos.load(markowitz_id)
    
    # 加载资产收益率
    min_date = df_pos.index.min()
    #max_date = df_pos.index.max()
    max_date = (datetime.now() - timedelta(days=1)) # yesterday


    data = {}
    for asset_id in df_pos.columns:
        data[asset_id] = load_nav_series(asset_id, min_date, max_date)
    df_nav = pd.DataFrame(data).fillna(method='pad')
    df_inc  = df_nav.pct_change().fillna(0.0)

    # 计算复合资产净值
    df_nav_portfolio = DFUtil.portfolio_nav(df_inc, df_pos, result_col='portfolio')

    df_result = df_nav_portfolio[['portfolio']].rename(columns={'portfolio':'mz_nav'}).copy()
    df_result.index.name = 'mz_date'
    df_result['mz_inc'] = df_result['mz_nav'].pct_change().fillna(0.0)
    df_result['mz_markowitz_id'] = markowitz['globalid']
    df_result = df_result.reset_index().set_index(['mz_markowitz_id', 'mz_date'])

    asset_mz_markowitz_nav.save(markowitz_id, df_result)

