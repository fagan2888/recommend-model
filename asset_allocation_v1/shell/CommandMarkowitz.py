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
from db import database, asset_mz_markowitz, asset_mz_markowitz_asset, asset_mz_markowitz_criteria, asset_mz_markowitz_nav, asset_mz_markowitz_pos, asset_mz_markowitz_sharpe
from db import asset_ra_pool, asset_ra_pool_nav, asset_rs_reshape, asset_rs_reshape_nav, asset_rs_reshape_pos
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates
from util import xdict

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)  
@click.option('--full/--no-full', 'optfull', default=False, help=u'include all instance')
@click.option('--id', 'optid', help=u'specify markowitz id')
@click.option('--name', 'optname', default=None, help=u'specify markowitz name')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace pool if exists')
@click.option('--start-date', 'startdate', default='2012-07-27', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.option('--lookback', type=int, default=26, help=u'howmany weeks to lookback')
@click.option('--adjust-period', type=int, default=1, help=u'adjust every how many weeks')
@click.option('--turnover', 'optturnover', type=float, default=0, help=u'fitler by turnover')
@click.option('--bootstrap/--no-bootstrap', 'optbootstrap', default=True, help=u'use bootstrap or not')
@click.option('--bootstrap-count', 'optbootcount', type=int, default=0, help=u'use bootstrap or not')
@click.option('--cpu-count', 'optcpu', type=int, default=0, help=u'how many cpu to use, (0 for all available)')
@click.option('--short-cut', type=click.Choice(['high', 'low', 'default']))
@click.option('--assets', multiple=True, help=u'assets')
@click.pass_context
def markowitz(ctx, optfull, optid, optname, opttype, optreplace, startdate, enddate, lookback, adjust_period, optturnover, optbootstrap, optbootcount, optcpu, short_cut, assets):

    '''markowitz group
    '''
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        if optfull is False:
            ctx.invoke(allocate, optid=optid, optname=optname, opttype=opttype, optreplace=optreplace, startdate=startdate, enddate=enddate, lookback=lookback, adjust_period=adjust_period, turnover=optturnover, optbootstrap=optbootstrap, optbootcount=optbootcount, optcpu=optcpu, short_cut=short_cut, assets=assets)
            ctx.invoke(nav, optid=optid)
            ctx.invoke(turnover, optid=optid)
        else:
            ctx.invoke(nav, optid=optid)
            ctx.invoke(turnover, optid=optid)
    else:
        # click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
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
            s = 'markowitz instance [%s] existed' % str(optid)
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
@click.option('--name', 'optname', default=None, help=u'specify markowitz name')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace pool if exists')
@click.option('--start-date', 'startdate', default='2012-07-27', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.option('--lookback', type=int, default=26, help=u'howmany weeks to lookback')
@click.option('--adjust-period', type=int, default=1, help=u'adjust every how many weeks')
@click.option('--turnover', type=float, default=0, help=u'fitler by turnover')
@click.option('--bootstrap/--no-bootstrap', 'optbootstrap', default=True, help=u'use bootstrap or not')
@click.option('--bootstrap-count', 'optbootcount', type=int, default=200, help=u'use bootstrap or not')
@click.option('--cpu-count', 'optcpu', type=int, default=0, help=u'how many cpu to use, (0 for all available)')
@click.option('--short-cut', type=click.Choice(['default', 'high', 'low']))
@click.option('--algo', 'optalgo', type=click.Choice(['markowitz', 'average']), help=u'which algorithm to use for allocate')
@click.argument('assets', nargs=-1)
@click.pass_context
def allocate(ctx, optid, optname, opttype, optreplace, startdate, enddate, lookback, adjust_period, turnover,  optbootstrap, optbootcount, optcpu, short_cut, optalgo, assets):
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
            s = 'markowitz instance [%s] existed' % str(optid)
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
        if short_cut == 'high':
            assets = {
                120000001:  {'sum1': 0,    'sum2' : 0,   'upper': 0.70,  'lower': 0.0}, #沪深300指数修型
                120000002:  {'sum1': 0,    'sum2' : 0,   'upper': 0.70,  'lower': 0.0}, #中证500指数修型
                120000013:  {'sum1': 0.65, 'sum2' : 0,   'upper': 0.35, 'lower': 0.0}, #标普500指数
                120000015:  {'sum1': 0.65, 'sum2' : 0,   'upper': 0.35, 'lower': 0.0}, #恒生指数修型
                120000014:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.35, 'lower': 0.0}, #黄金指数修型
                # 120000029:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.20, 'lower': 0.0}, #南华商品指数
                # 120000028:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.20, 'lower': 0.0}, #标普高盛原油商品指数收益率
                # 120000031:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.20, 'lower': 0.0}, #房地产指数
            }
            if optname is None:
                optname = u'马克维茨%s(高风险)' % today.strftime("%m%d")
        elif short_cut == 'low':
            assets = {
                #120000010:  {'sum1': 0, 'sum2': 0, 'upper': 1.0, 'lower': 0.0},
                120000011:  {'sum1': 0, 'sum2': 0, 'upper': 1.0, 'lower': 0.0},
            }
            if optname is None:
                optname = u'马克维茨%s(低风险)' % today.strftime("%m%d")
            if optalgo is None:
                optalgo = 'average'
                
        else: # short_cut == 'default'
            assets = {
                120000001:  {'sum1': 0,    'sum2' : 0,   'upper': 1.0,  'lower': 0.0}, #沪深300指数修型
                120000002:  {'sum1': 0,    'sum2' : 0,   'upper': 1.0,  'lower': 0.0}, #中证500指数修型

                # 120000013:  {'sum1': 0.65, 'sum2' : 0,   'upper': 0.35, 'lower': 0.0}, #标普500指数
                # 120000015:  {'sum1': 0.65, 'sum2' : 0,   'upper': 0.35, 'lower': 0.0}, #恒生指数修型
                # 120000014:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.35, 'lower': 0.0}, #黄金指数修型
                # 120000029:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.20, 'lower': 0.0}, #南华商品指数
                # 120000028:  {'sum1': 0.65, 'sum2' : 0.45,'upper': 0.20, 'lower': 0.0}, #标普高盛原油商品指数收益

                120000013:  {'sum1': 0, 'sum2' : 0,   'upper': 0.55, 'lower': 0.0}, #标普500指数
                120000015:  {'sum1': 0, 'sum2' : 0,   'upper': 0.55, 'lower': 0.0}, #恒生指数修型
                120000014:  {'sum1': 0, 'sum2' : 0,'upper': 0.32, 'lower': 0.0}, #黄金指数修型
                120000029:  {'sum1': 0, 'sum2' : 0,'upper': 0.20, 'lower': 0.0}, #南华商品指数
                120000028:  {'sum1': 0, 'sum2' : 0,'upper': 0.15, 'lower': 0.0}, #标普高盛原油商品指数收益率
                120000031:  {'sum1': 0, 'sum2' : 0,'upper': 0.20, 'lower': 0.0}, #房地产指数

            }

    if optname is None:
        optname = u'马克维茨%s(实验)' % today.strftime("%m%d")

    bootstrap = optbootcount if optbootstrap else None

    if optalgo == 'average':
        df = average_days(startdate, enddate, assets)
    else:
        df = markowitz_days(
            startdate, enddate, assets,
            label='markowitz', lookback=lookback, adjust_period=adjust_period, bootstrap=bootstrap, cpu_count=optcpu)

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

    #
    # 导入数据: markowitz_asset
    #
    assets = {k: merge_asset_name_and_type(k, v) for (k, v) in assets.iteritems()}
    df_asset = pd.DataFrame(assets).T
    df_asset.index.name = 'mz_markowitz_asset_id'
    df_asset['mz_markowitz_id'] = optid
    df_asset.rename(inplace=True, columns={
        'upper':  'mz_upper_limit', 'lower':'mz_lower_limit',
        'sum1':'mz_sum1_limit',  'sum2': 'mz_sum2_limit'
    })
    df_asset = df_asset.reset_index().set_index(['mz_markowitz_id', 'mz_markowitz_asset_id'])
    asset_mz_markowitz_asset.save(optid, df_asset)

    #
    # 导入数据: markowitz_pos
    #
    df = df.round(4)             # 四舍五入到万分位

    #每四周做平滑
    smooth = 4
    if bootstrap < 100:
        smooth = 1
    elif bootstrap >= 100 and bootstrap < 200:
        smooth = 2
    elif bootstrap >= 200:
        smooth = 4

    print bootstrap, smooth

    df = df.rolling(window = smooth, min_periods = 1).mean()

    df[df.abs() < 0.0009999] = 0 # 过滤掉过小的份额
    df = df.apply(npu.np_pad_to, raw=True, axis=1) # 补足缺失
    df = DFUtil.filter_same_with_last(df)          # 过滤掉相同
    if turnover >= 0.01:
        df = DFUtil.filter_by_turnover(df, turnover)   # 基于换手率进行规律 

    df.index.name = 'mz_date'
    df.columns.name='mz_markowitz_asset'

    # 计算原始资产仓位
    raw_ratios = {}
    raw_assets = {}
    for asset_id in df.columns:
        if asset_id / 10000000 != 4:
            raw_assets[asset_id] = asset_id
            raw_ratios[asset_id] = df[asset_id]
        else:
            #
            # 修型资产
            #
            rs_reshape = asset_rs_reshape.find(asset_id)
            if rs_reshape is None:
                raw_assets[asset_id] = asset_id
                raw_ratios[asset_id] = df[asset_id]
            else:
                raw_assets[asset_id] = rs_reshape['rs_asset_id']
                sr_reshape_pos = asset_rs_reshape_pos.load_series(asset_id, reindex=df.index)
                raw_ratios[asset_id] = df[asset_id] * sr_reshape_pos
    df_raw_ratio = pd.DataFrame(raw_ratios, columns=df.columns)
    df_raw_asset = pd.DataFrame(raw_assets, index=df.index, columns=df.columns)

    df_tosave = pd.concat({'mz_markowitz_ratio': df, 'mz_asset_id':df_raw_asset, 'mz_ratio': df_raw_ratio}, axis=1)

    # index
    df_tosave['mz_markowitz_id'] = optid
    df_tosave = df_tosave.reset_index().set_index(['mz_markowitz_id', 'mz_date'])

    # unstack
    df_tosave = df_tosave.stack()
    df_tosave = df_tosave.loc[(df_tosave['mz_ratio'] > 0) | (df_tosave['mz_markowitz_ratio'] > 0)]
    
    # save
    # print df_tosave
    asset_mz_markowitz_pos.save(optid, df_tosave)

    # 导入数据: markowitz_criteria
    criterias = {'return': '0001', 'risk':'0002', 'sharpe':'0003'}
    df_sharpe.index.name = 'mz_date'
    for column in df_sharpe.columns:
        criteria_id = criterias[column]
        df_criteria = df_sharpe[column].to_frame('mz_value')
        df_criteria['mz_markowitz_id'] = optid
        df_criteria['mz_criteria_id'] = criteria_id
        df_criteria = df_criteria.reset_index().set_index(['mz_markowitz_id', 'mz_criteria_id', 'mz_date'])
        asset_mz_markowitz_criteria.save(optid, criteria_id,  df_criteria)
    
    click.echo(click.style("markowitz allocation complement! instance id [%s]" % (optid), fg='green'))

    #
    # 在ctx中记录markowitz id 以便命令链的后面使用
    #
    ctx.obj['markowitz'] = optid
    if short_cut == 'high' or short_cut == 'default':
        ctx.obj['markowitz.high'] = optid
    if short_cut == 'low':
        ctx.obj['markowitz.low'] = optid

    return 0

def parse_asset(asset):
    segments = [s.strip() for s in asset.strip().split(':')]

    if len(segments) == 1:
        result = (int(segments[0]), {
            'upper': 1.0, 'lower': 0.0, 'sum1': 0, 'sum2': 0})
    elif len(segments) == 2:
        result = (int(segments[0]), {
            'upper': float(segments[1]), 'lower': 0.0, 'sum1': 0, 'sum2': 0})
    elif len(segments) == 3:
        result = (int(segments[0]), {
            'upper': float(segments[1]), 'lower': float(segments[2]), 'sum1': 0, 'sum2': 0})
    elif len(segments) == 4:
        result = (int(segments[0]), {
            'upper': float(segments[1]), 'lower': float(segments[2]), 'sum1': float(segments[3]), 'sum2': 0})
    else:
        if len(segments) >= 5:
            result = (int(segments[0]), {
                'upper': float(segments[1]), 'lower': float(segments[2]), 'sum1': float(segments[3]), 'sum2': float(segments[4]) })
        else:
            result = (None, {'upper': 1.0, 'lower': 0.0, 'sum1': 0, 'sum2': 0})

    return result

def merge_asset_name_and_type(asset_id, asset_data):
    xtype = asset_id / 10000000

    if xtype == 4:
        #
        # 修型资产
        #
        asset = asset_rs_reshape.find(asset_id)
        (name, category, raw_asset) = (asset['rs_name'], asset['rs_asset'], asset['rs_asset_id'])
        (raw_name, ph) = database.load_asset_name_and_type(raw_asset)
    else:
        (name, category) = database.load_asset_name_and_type(asset_id)
        (raw_asset, raw_name) = (asset_id, name)
        
    return xdict.merge(asset_data, {
        'mz_asset_id': raw_asset,
        'mz_asset_name': raw_name,
        'mz_markowitz_asset_name': name,
        'mz_asset_type': category,
    })

def average_days(start_date, end_date, assets):
    '''perform markowitz asset for days
    '''
    
    if len(assets) > 0:
        ratio = 1.0 / len(assets)
    else:
        ratio = 0

    data = {k: {start_date: 0} for k in ['return', 'risk', 'sharpe']}
    
    data.update({k: {start_date: ratio} for k in assets.keys()})

    df = pd.DataFrame(data)

    return df

def markowitz_days(start_date, end_date, assets, label, lookback, adjust_period, bootstrap, cpu_count=0):
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
    s = 'perform %-12s' % label
    data = {}
    with click.progressbar(
            adjust_index, label=s,
            item_show_func=lambda x:  x.strftime("%Y-%m-%d") if x else None) as bar:
        for day in bar:
            # bar.update(1)
            logger.debug("%s : %s", s, day.strftime("%Y-%m-%d"))
            # 高风险资产配置
            data[day] = markowitz_day(day, lookback, assets, bootstrap, cpu_count)

    return pd.DataFrame(data).T

def markowitz_day(day, lookback, assets, bootstrap, cpu_count):
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
        data[asset] = load_nav_series(asset, index, begin_date, end_date)
    df_nav = pd.DataFrame(data).fillna(method='pad')
    df_inc  = df_nav.pct_change().fillna(0.0)

    return markowitz_r(df_inc, assets, bootstrap, cpu_count)

def markowitz_r(df_inc, limits, bootstrap, cpu_count):
    '''perform markowitz
    '''
    bound = []
    for asset in df_inc.columns:
        bound.append(limits[asset])

    if bootstrap is None:
        risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
    else:
        risk, returns, ws, sharpe = PF.markowitz_bootstrape(df_inc, bound, cpu_count=cpu_count, bootstrap_count=bootstrap)

    sr_result = pd.concat([
        pd.Series(ws, index=df_inc.columns),
        pd.Series((sharpe, risk, returns), index=['sharpe','risk', 'return'])
    ])

    return sr_result

def load_nav_series(asset_id, reindex=None, begin_date=None, end_date=None):
    xtype = asset_id / 10000000

    if xtype == 1:
        #
        # 基金池资产
        #
        asset_id %= 10000000
        (pool_id, category) = (asset_id / 100, asset_id % 100)
        ttype = pool_id / 10000
        sr = asset_ra_pool_nav.load_series(
            pool_id, category, ttype, reindex=reindex, begin_date=begin_date, end_date=end_date)
    elif xtype == 3:
        #
        # 基金池资产
        #
        sr = base_ra_fund_nav.load_series(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
    elif xtype == 4:
        #
        # 修型资产
        #
        sr = asset_rs_reshape_nav.load_series(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
    elif xtype == 12:
        #
        # 指数资产
        #
        sr = base_ra_index_nav.load_series(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
    else:
        sr = pd.Series()

    return sr

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
        if 'markowitz' in ctx.obj:
            markowitzs = [str(ctx.obj['markowitz'])]
        else:
            markowitzs = None

    df_markowitz = asset_mz_markowitz.load(markowitzs)

    if optlist:
        df_markowitz['mz_name'] = df_markowitz['mz_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_markowitz, headers='keys', tablefmt='psql')
        return 0
    
    with click.progressbar(
            df_markowitz.iterrows(), len(df_markowitz.index), label='%-20s' % 'update nav',
            item_show_func=lambda x:  str(x[1]['globalid']) if x else None) as bar:
        for _, markowitz in bar:
            # bar.update(1)
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
        data[asset_id] = load_nav_series(asset_id, begin_date=min_date, end_date=max_date)
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

@markowitz.command()
@click.option('--id', 'optid', help=u'ids of markowitz to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.pass_context
def turnover(ctx, optid, optlist):
    ''' calc pool turnover and inc
    '''
    if optid is not None:
        markowitzs = [s.strip() for s in optid.split(',')]
    else:
        if 'markowitz' in ctx.obj:
            markowitzs = [str(ctx.obj['markowitz'])]
        else:
            markowitzs = None

    df_markowitz = asset_mz_markowitz.load(markowitzs)

    if optlist:

        df_markowitz['mz_name'] = df_markowitz['mz_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_markowitz, headers='keys', tablefmt='psql')
        return 0
    
    data = []
    with click.progressbar(
            df_markowitz.iterrows(), length=len(df_markowitz.index), label= '%-20s' % 'update turnover',
            item_show_func=lambda x:  str(x[1]['globalid']) if x else None) as bar:
        for _, markowitz in bar:
            # bar.update(1)
            turnover = turnover_update(markowitz)
            data.append((markowitz['globalid'], "%6.2f" % (turnover * 100)))

    headers = ['markowitz', 'turnover(%)']
    print(tabulate(data, headers=headers, tablefmt="psql"))                 
    # print(tabulate(data, headers=headers, tablefmt="fancy_grid"))                 
    # print(tabulate(data, headers=headers, tablefmt="grid"))                 
            
def turnover_update(markowitz):
    markowitz_id = markowitz['globalid']
    # 加载仓位信息
    df = asset_mz_markowitz_pos.load(markowitz_id, use_markowitz_ratio=False)

    # 计算宽口换手率
    sr_turnover = DFUtil.calc_turnover(df)

    criteria_id = 6
    df_result = sr_turnover.to_frame('mz_value')
    df_result['mz_markowitz_id'] = markowitz_id
    df_result['mz_criteria_id'] = criteria_id
    df_result = df_result.reset_index().set_index(['mz_markowitz_id', 'mz_criteria_id', 'mz_date'])
    asset_mz_markowitz_criteria.save(markowitz_id, criteria_id,  df_result)

    total_turnover = sr_turnover.sum()

    return total_turnover

    # df_result.reset_index(inplace=True)
    # df_result['turnover'] = df_result['turnover'].map(lambda x: "%6.2f%%" % (x * 100))
    # print tabulate(df_result, headers='keys', tablefmt='psql', stralign=u'right')
@markowitz.command()
@click.option('--id', 'optid', help=u'ids of markowitz to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.option('--exec/--no-exec', 'optexec', default=False, help=u'list instance to update')
@click.pass_context
def delete(ctx, optid, optlist, optexec):
    ''' delete markowitz instance
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

    if optid is None or not optexec:
         click.echo(click.style("\nboth --id and --exec is required to perform delete\n", fg='red'))
         return 0
    
    data = []
    with click.progressbar(
            df_markowitz.iterrows(), length=len(df_markowitz.index), label= '%-20s' % 'delete markowitz',
            item_show_func=lambda x:  str(x[1]['globalid']) if x else None) as bar:
        for _, markowitz in bar:
            perform_delete(markowitz)
            
def perform_delete(markowitz):
    markowitz_id = markowitz['globalid']

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    mz_markowitz        = Table('mz_markowitz', metadata, autoload=True)
    mz_markowitz_asset  = Table('mz_markowitz_asset', metadata, autoload=True)
    mz_markowitz_pos    = Table('mz_markowitz_pos', metadata, autoload=True)
    mz_markowitz_nav    = Table('mz_markowitz_nav', metadata, autoload=True)
    mz_markowitz_sharpe = Table('mz_markowitz_sharpe', metadata, autoload=True)

    #
    # 处理删除
    #

    mz_markowitz.delete(mz_markowitz.c.globalid == markowitz_id).execute()
    mz_markowitz_asset.delete(mz_markowitz_asset.c.mz_markowitz_id == markowitz_id).execute()
    mz_markowitz_pos.delete(mz_markowitz_pos.c.mz_markowitz_id == markowitz_id).execute()
    mz_markowitz_nav.delete(mz_markowitz_nav.c.mz_markowitz_id == markowitz_id).execute()
    mz_markowitz_sharpe.delete(mz_markowitz_sharpe.c.mz_markowitz_id == markowitz_id).execute()
    mz_markowitz.delete(mz_markowitz.c.globalid == markowitz_id).execute()

# @markowitz.command()
# @click.option('--id', 'optid', help=u'ids of markowitz to update')
# @click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
# @click.pass_context
# def maxdd(ctx, optid, optlist):
#     ''' delete markowitz instance
#     '''
#     if optid is not None:
#         markowitzs = [s.strip() for s in optid.split(',')]
#     else:
#         markowitzs = None

#     df_markowitz = asset_mz_markowitz.load(markowitzs)

#     if optlist:

#         df_markowitz['mz_name'] = df_markowitz['mz_name'].map(lambda e: e.decode('utf-8'))
#         print tabulate(df_markowitz, headers='keys', tablefmt='psql')
#         return 0

#     data = []
#     for _, markowitz in df_markowitz.iterrows():
#         perform_maxdd(markowitz)
            
# def perform_maxdd(markowitz):
#     markowitz_id = markowitz['globalid']
#     sdate = '2012-07-27'
#     tdates = base_trade_dates.load_index(sdate);
#     # sr_nav = database.load_nav_series(markowitz_id, reindex=tdates, begin_date=sdate)
#     sr_nav = asset_mz_markowitz_nav.load_series(markowitz_id, reindex=tdates, begin_date=sdate)

#     positive, total = (0, 0)
#     for w in [60, 120, 250]:
#         tmp = sr_nav.rolling(window=250, min_periods=250).apply(maxdd);
#         tmp = tmp.dropna()
#         print "win", w,  tmp.sum()/len(tmp)

# def maxdd(x):
#     y = x[-1]/x[0] - 1
#     max_drawdown = (x/np.maximum.accumulate(x) - 1).min()
#     if (y / abs(max_drawdown)) > 2:
#         return 1
#     return 0
    
