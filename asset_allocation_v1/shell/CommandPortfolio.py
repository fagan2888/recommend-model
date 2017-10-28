#coding=utf8

import pdb
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
import EqualRiskAssetRatio
import EqualRiskAsset
import HighLowRiskAsset
import os
import DBData
import AllocationData
import time
import RiskHighLowRiskAsset
import ModelHighLowRisk
import GeneralizationPosition
import Const
import WeekFund2DayNav
import FixRisk
import DFUtil
import LabelAsset
import util_numpy as npu
import TradeNav

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from tabulate import tabulate
from db import *
from util.xdebug import dd

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)  
@click.option('--full/--no-full', 'optfull', default=False, help=u'include all instance')
@click.option('--new/--no-new', 'optnew', default=False, help=u'use new framework')
@click.option('--id', 'optid', help=u'specify portfolio id')
@click.option('--name', 'optname', default=None, help=u'specify portfolio name')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace pool if exists')
@click.option('--ratio', 'optratio', type=int, default=None, help=u'specified which ratio_id to use')
@click.option('--pool', 'optpool', default=0, help=u'which pool to use for each asset (eg. 120000001:11110100,120000002:11110100')
@click.option('--risk', 'optrisk', default='10,1,2,3,4,5,6,7,8,9', help=u'which risk to calc, [1-10]')
@click.option('--turnover', 'optturnover',  type=float, default=0, help=u'fitler by turnover')
@click.option('--end-date', 'optenddate', default=None, help=u'calc end date for nav')
@click.pass_context
def portfolio(ctx, optfull, optnew, optid, optname, opttype, optreplace, optratio, optpool, optrisk, optturnover,optenddate):

    '''generate final portolio
    '''
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        if optnew:
            ctx.invoke(pos, optid=optid, optrisk=optrisk)
            ctx.invoke(nav, optid=optid, optrisk=optrisk, optenddate=optenddate)
            ctx.invoke(turnover, optid=optid)
        else:
            if optfull is False:
                if optid is not None:
                    tmpid = int(optid)
                else:
                    tmpid = optid
                ctx.invoke(allocate, optid=tmpid, optname=optname, opttype=opttype, optreplace=optreplace, optratio=optratio, optpool=optpool, optrisk=optrisk, turnover=optturnover)
                ctx.invoke(nav, optid=optid, optrisk=optrisk, optenddate=optenddate)
                ctx.invoke(turnover, optid=optid)
            else:
                ctx.invoke(nav, optid=optid, optrisk=optrisk, optenddate=optenddate)
                ctx.invoke(turnover, optid=optid)
    else:
        # click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        pass

@portfolio.command()
@click.option('--id', 'optid', type=int, help=u'specify portfolio id')
@click.option('--name', 'optname', default=None, help=u'specify portfolio name')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace portfolio if exists')
@click.option('--ratio', 'optratio', type=int, default=None, help=u'specified which ratio_id to use')
@click.option('--pool', 'optpool', default=0, help=u'which pool to use for each asset (eg. 120000001:11110100,120000002:11110200')
@click.option('--risk', 'optrisk', default='10,1,2,3,4,5,6,7,8,9', help=u'which risk to calc, [1-10]')
@click.option('--turnover', type=float, default=0, help=u'fitler by turnover')
@click.pass_context
def allocate(ctx, optid, optname, opttype, optreplace, optratio, optpool, optrisk, turnover):
    '''generate final portfolio
    '''

    if optratio is None:
        if 'highlow' not in ctx.obj:
            click.echo(click.style("--ratio is required, aborted!", fg="red"))
            return 0
        
        optratio = ctx.obj['highlow']
    #
    # 处理id参数
    #
    today = datetime.now()
    if optid is not None:
        #
        # 检查id是否存在
        #
        df_existed = asset_ra_portfolio.load([str(optid * 10 + x) for x in range(0, 10)])
        if not df_existed.empty:
            s = 'portfolio instance [%s] existed' % str(optid)
            if optreplace:
                click.echo(click.style("%s, will replace!" % s, fg="yellow"))
            else:
                click.echo(click.style("%s, import aborted!" % s, fg="red"))
                return -1;
    else:
        #
        # 自动生成id
        #
        prefix = '80' + today.strftime("%m%d");
        between_min, between_max = ('%s00' % (prefix), '%s99' % (prefix))

        max_id = asset_ra_portfolio.max_id_between(between_min, between_max)
        if max_id is None:
            optid = int(between_min)
        else:
            if int(max_id) >= int(between_max):
                if optreplace:
                    s = "run out of instance id [%s]" % max_id
                    click.echo(click.style("%s, will replace!" % s, fg="yellow"))
                else:
                    s = "run out of instance id [%s]" % max_id
                    click.echo(click.style("%s, aborted!" % s, fg="red"))
                    return -1

            if optreplace:
                optid = int(max_id)
            else:
                optid = int(max_id) + 10

    if optname is None:
        optname = u'智能组合%s' % today.strftime("%m%d")

    #
    # 加载用到的资产
    #
    df_asset = database.load_asset_and_pool(optratio)
    
    df_asset = df_asset.rename(columns={
        'asset_id': 'ra_asset_id',
        'asset_name':'ra_asset_name',
        'asset_type':'ra_asset_type',
        'pool_id': 'ra_pool_id',
    })

    if '11310100' not in df_asset['ra_asset_id'].values:
        sr = ('11310100', '货币资产', 31, '11310100')
        df_asset.ix[len(df_asset.index)] = sr
        
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    ra_portfolio        = Table('ra_portfolio', metadata, autoload=True)
    ra_portfolio_alloc  = Table('ra_portfolio_alloc', metadata, autoload=True)
    ra_portfolio_asset  = Table('ra_portfolio_asset', metadata, autoload=True)
    ra_portfolio_pos    = Table('ra_portfolio_pos', metadata, autoload=True)
    ra_portfolio_nav    = Table('ra_portfolio_nav', metadata, autoload=True)

    #
    # 处理替换
    #
    if optreplace:
        ra_portfolio_nav.delete(ra_portfolio_nav.c.ra_portfolio_id.between(optid, optid + 9)).execute()
        ra_portfolio_pos.delete(ra_portfolio_pos.c.ra_portfolio_id.between(optid, optid + 9)).execute()
        ra_portfolio_asset.delete(ra_portfolio_asset.c.ra_portfolio_id == optid).execute()
        ra_portfolio_alloc.delete(ra_portfolio_alloc.c.ra_portfolio_id == optid).execute()
        ra_portfolio.delete(ra_portfolio.c.globalid == optid).execute()

    now = datetime.now()
    # 导入数据: portfolio
    row = {
        'globalid': optid, 'ra_type':opttype, 'ra_name': optname,
        'ra_algo': 1, 'ra_ratio_id': optratio,
        'ra_persistent': 0, 'created_at': func.now(), 'updated_at': func.now()
    }
    ra_portfolio.insert(row).execute()

    #
    # 计算每个风险的配置
    #
    with click.progressbar(
            database.load_alloc_and_risk(optratio),
            label=('update %-13s' % 'portfolio').ljust(30),
            item_show_func=lambda x:  'risk %d' % int(x[0] * 10) if x else None) as bar:
        for (risk, ratio_id) in bar:
            gid = optid + (int(risk * 10) % 10)
            name = optname + u"-等级%d" % int(risk * 10)

            # 加载资产配置比例
            df_ratio = database.load_pos_frame(ratio_id)
            #print df_ratio
            # print df_ratio.sum(axis=1)
            if '11310100' not in df_ratio.columns:
                df_ratio['11310100'] = 1 - df_ratio.sum(axis=1)
            else:
                df_ratio['11310100'] += 1 - df_ratio.sum(axis=1)
            # print df_ratio.head()

            start = df_ratio.index.min()
            index = df_ratio.index.copy()
            #
            # 加载基金池
            #
            pools = {}
            for _, row in df_asset.iterrows():
                fund = asset_ra_pool_fund.load(row['ra_pool_id'])
                if not fund.empty:
                    index = index.union(fund.index.get_level_values(0)).unique()
                pool = (row['ra_pool_id'], fund[['ra_fund_code', 'ra_fund_type']])
                pools[row['ra_asset_id']] = pool
            else:
                if '11310100' not in pools:
                    fund = asset_ra_pool_fund.load('11310100')
                    if not fund.empty:
                        index = index.union(fund.index.get_level_values(0)).unique()
                    pool = ('11310100', fund[['ra_fund_code', 'ra_fund_type']])
                    pools['11310100'] = pool

            #
            # 根据基金池和配置比例的索引并集reindex数据
            #
            index = index[index >= start]
            df_ratio = df_ratio.reindex(index, method='pad')
            tmp = {}
            for k, v in pools.iteritems():
                (pool, df_fund) = v
                tmp[k] = (pool, df_fund.unstack().reindex(index, method='pad').stack())
            pools = tmp
            #
            # 计算基金配置比例
            #
            data = []
            for day, row in df_ratio.iterrows():
                for asset_id, ratio in row.iteritems():
                    if (ratio <= 0):
                        continue
                    # 选择基金
                    (pool_id, df_fund) = pools[asset_id]
                    segments = choose_fund_avg(day, pool_id, ratio, df_fund.loc[day])
                    #if int(risk * 10) == 1:
                    #    print segments
                    data.extend(segments)
            #print data
            df_raw = pd.DataFrame(data, columns=['ra_date', 'ra_pool_id', 'ra_fund_id', 'ra_fund_code', 'ra_fund_type', 'ra_fund_ratio'])
            df_raw.set_index(['ra_date', 'ra_pool_id', 'ra_fund_id'], inplace=True)

            #
            # 导入数据: portfolio_alloc
            #
            row = {
                'globalid': gid, 'ra_type':opttype, 'ra_name': name,
                'ra_portfolio_id': optid, 'ra_ratio_id': ratio_id, 'ra_risk': risk,
                'created_at': func.now(), 'updated_at': func.now()
            }
            ra_portfolio_alloc.insert(row).execute()

            #
            # 导入数据: portfolio_pos
            #
            #print df_raw.head()
            df_raw.loc[df_raw['ra_fund_ratio'] < 0.00009999, 'ra_fund_ratio'] = 0 # 过滤掉过小的份额
            df_raw['ra_fund_ratio'] = df_raw['ra_fund_ratio'].round(4)            # 四舍五入到万分位

            df_tmp = df_raw[['ra_fund_ratio']]

            #print gid, df_tmp
            df_tmp = df_tmp.unstack([1, 2])
            df_tmp = df_tmp.apply(npu.np_pad_to, raw=True, axis=1) # 补足缺失
            df_tmp = DFUtil.filter_same_with_last(df_tmp)          # 过滤掉相同
            if turnover >= 0.01:
                df_tmp = DFUtil.filter_by_turnover(df_tmp, turnover)   # 基于换手率进行规律
                df_tmp.index.name = 'ra_date'
            df_tmp = df_tmp.stack([1, 2])
            df = df_tmp.merge(df_raw[['ra_fund_code', 'ra_fund_type']], how='left', left_index=True, right_index=True)


            # index
            df['ra_portfolio_id'] = gid
            df = df.reset_index().set_index(['ra_portfolio_id', 'ra_date', 'ra_pool_id', 'ra_fund_id'])
            df_tosave = df.loc[(df['ra_fund_ratio'] > 0)].copy()

            # save
            # print df_tosave
            asset_ra_portfolio_pos.save(gid, df_tosave)

            #
            # 导入数据: portfolio_asset
            #
            pool_ids = df.index.levels[2]
            df_asset_tosave = df_asset[df_asset['ra_pool_id'].isin(pool_ids)].copy()
            df_asset_tosave['ra_portfolio_id'] = gid
            df_asset_tosave = df_asset_tosave.set_index(['ra_portfolio_id', 'ra_asset_id'])
            asset_ra_portfolio_asset.save([gid], df_asset_tosave)


        # click.echo(click.style("portfolio allocation complement! instance id [%s]" % (gid), fg='green'))
    #
    # 在context中保存optid
    #
    ctx.obj['portfolio'] = optid
    
    click.echo(click.style("portfolio allocation complement! instance id [%s]" % (optid), fg='green'))

def choose_fund_avg(day, pool_id, ratio, df_fund):
    # 比例均分
    if not df_fund.empty:
        fund_ratio = ratio / len(df_fund)
    else:
        fund_ratio = 0

    return [(day, pool_id, fund_id, x['ra_fund_code'], x['ra_fund_type'], fund_ratio) for fund_id, x in df_fund.iterrows()]

@portfolio.command()
@click.option('--id', 'optid', help=u'ids of portfolio to update')
@click.option('--type', 'opttype', default='8,9', help=u'which type to run')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.option('--risk', 'optrisk', default='10,1,2,3,4,5,6,7,8,9', help=u'which risk to calc, [1-10]')
@click.pass_context
def pos(ctx, optid, opttype, optlist, optrisk):
    ''' calc pool nav and inc
    '''
    if optid is not None:
        portfolios = [s.strip() for s in optid.split(',')]
    else:
        if 'portfolio' in ctx.obj:
            portfolios = [str(ctx.obj['portfolio'])]
        else:
            portfolios = None

    xtypes = [s.strip() for s in opttype.split(',')]

    df_portfolio = asset_ra_portfolio.load(portfolios, xtypes)

    if optlist:
        df_portfolio['ra_name'] = df_portfolio['ra_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_portfolio, headers='keys', tablefmt='psql')
        return 0

    for _, portfolio in df_portfolio.iterrows():
        pos_update_alloc(portfolio, optrisk)
        
def pos_update_alloc(portfolio, optrisk):
    risks = [int(s.strip()) for s in optrisk.split(',')]
    df_alloc = asset_ra_portfolio_alloc.where_portfolio_id(portfolio['globalid'], risks)
    df_alloc = df_alloc.loc[(df_alloc['ra_risk'] * 10).astype(int).isin(risks)]
    
    with click.progressbar(
            df_alloc.iterrows(), length=len(df_alloc.index),
            label=('update pos %-9s' % (portfolio['globalid'])).ljust(30),
            item_show_func=lambda x: str(x[1]['globalid']) if x else None) as bar:
        for _, alloc in bar:
            pos_update(portfolio, alloc)

    click.echo(click.style("portfolio allocation complement! instance id [%s]" % (portfolio['globalid']), fg='green'))
        
def pos_update(portfolio, alloc):
    gid = alloc['globalid']

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    ra_portfolio_criteria = Table('ra_portfolio_criteria', metadata, autoload=True)
    ra_portfolio_contrib = Table('ra_portfolio_contrib', metadata, autoload=True)
    ra_portfolio_pos    = Table('ra_portfolio_pos', metadata, autoload=True)
    ra_portfolio_nav    = Table('ra_portfolio_nav', metadata, autoload=True)

    #
    # 处理替换
    #
    ra_portfolio_criteria.delete(ra_portfolio_criteria.c.ra_portfolio_id == gid).execute()
    ra_portfolio_contrib.delete(ra_portfolio_contrib.c.ra_portfolio_id == gid).execute()
    ra_portfolio_nav.delete(ra_portfolio_nav.c.ra_portfolio_id == gid).execute()
    ra_portfolio_pos.delete(ra_portfolio_pos.c.ra_portfolio_id == gid).execute()

    #
    # 加载参数
    #
    df_argv = asset_ra_portfolio_argv.load([gid])
    df_argv.reset_index(level=0, inplace=True)
    argv = df_argv['ra_value'].to_dict()

    # lookback = int(argv.get('lookback', '26'))
    # adjust_period = int(argv.get('adjust_period', 1))
    # wavelet_filter_num = int(argv.get('optwaveletfilternum', 2))
    turnover = float(argv.get('turnover', 0.4))

    # algo = alloc['ra_algo'] if alloc['ra_algo'] != 0 else portfolio['ra_algo']
    algo = portfolio['ra_algo']
    
    if algo == 1:
        #
        # 等权均分
        #
        df_raw = kun(portfolio, alloc)
    else:
        click.echo(click.style("\n unknow algo %d for %s\n" % (algo, gid), fg='red'))
        return

    df_tmp = df_raw[['ra_fund_ratio']]

    #print gid, df_tmp
    df_tmp = df_tmp.unstack([1, 2])
    df_tmp = df_tmp.apply(npu.np_pad_to, raw=True, axis=1) # 补足缺失
    df_tmp = DFUtil.filter_same_with_last(df_tmp)          # 过滤掉相同
    if turnover >= 0.01:
        df_tmp = DFUtil.filter_by_turnover(df_tmp, turnover)   # 基于换手率进行规律
        df_tmp.index.name = 'ra_date'
    df_tmp = df_tmp.stack([1, 2])
    df = df_tmp.merge(df_raw[['ra_fund_code', 'ra_fund_type']], how='left', left_index=True, right_index=True)

    # index
    df['ra_portfolio_id'] = gid
    df = df.reset_index().set_index(['ra_portfolio_id', 'ra_date', 'ra_pool_id', 'ra_fund_id'])
    df_tosave = df.loc[(df['ra_fund_ratio'] > 0)].copy()

    # save
    # print df_tosave
    asset_ra_portfolio_pos.save(gid, df_tosave)

def kun(portfolio, alloc):
    gid = alloc['globalid']
    risk = int(alloc['ra_risk'] * 10)
    ratio_id = alloc['ra_ratio_id']
    
    #
    # 加载用到的资产池
    #
    df_asset = asset_ra_portfolio_asset.load([gid])
    
    if '11310100' not in df_asset['ra_asset_id'].values:
        sr = (gid, '11310100', '货币资产', 31, '11310100')
        df_asset.ix[len(df_asset.index)] = sr

    # 加载资产配置比例
    df_ratio = database.load_pos_frame(ratio_id)
    #print df_ratio
    # print df_ratio.sum(axis=1)
    if '11310100' not in df_ratio.columns:
        df_ratio['11310100'] = 1 - df_ratio.sum(axis=1)
    else:
        df_ratio['11310100'] += 1 - df_ratio.sum(axis=1)
    # print df_ratio.head()

    start = df_ratio.index.min()
    index = df_ratio.index.copy()

    #
    # 加载基金池
    #
    pools = {}
    for _, row in df_asset.iterrows():
        fund = asset_ra_pool_fund.load(row['ra_pool_id'])
        if not fund.empty:
            index = index.union(fund.index.get_level_values(0)).unique()
        pool = (row['ra_pool_id'], fund[['ra_fund_code', 'ra_fund_type']])
        pools[row['ra_asset_id']] = pool
    else:
        if '11310100' not in pools:
            fund = asset_ra_pool_fund.load('11310100')
            if not fund.empty:
                index = index.union(fund.index.get_level_values(0)).unique()
            pool = ('11310100', fund[['ra_fund_code', 'ra_fund_type']])
            pools['11310100'] = pool

    #
    # 根据基金池和配置比例的索引并集reindex数据
    #
    index = index[index >= start]
    df_ratio = df_ratio.reindex(index, method='pad')
    tmp = {}
    for k, v in pools.iteritems():
        (pool, df_fund) = v
        tmp[k] = (pool, df_fund.unstack().reindex(index, method='pad').stack())
    pools = tmp
    #
    # 计算基金配置比例
    #
    data = []
    for day, row in df_ratio.iterrows():
        for asset_id, ratio in row.iteritems():
            if (ratio <= 0):
                continue
            # 选择基金
            (pool_id, df_fund) = pools[asset_id]
            segments = choose_fund_avg(day, pool_id, ratio, df_fund.loc[day])
            #if int(risk * 10) == 1:
            #    print segments
            data.extend(segments)
    #print data
    df_raw = pd.DataFrame(data, columns=['ra_date', 'ra_pool_id', 'ra_fund_id', 'ra_fund_code', 'ra_fund_type', 'ra_fund_ratio'])
    df_raw.set_index(['ra_date', 'ra_pool_id', 'ra_fund_id'], inplace=True)

    #
    # 导入数据: portfolio_pos
    #
    #print df_raw.head()
    df_raw.loc[df_raw['ra_fund_ratio'] < 0.00009999, 'ra_fund_ratio'] = 0 # 过滤掉过小的份额
    df_raw['ra_fund_ratio'] = df_raw['ra_fund_ratio'].round(4)            # 四舍五入到万分位

    return df_raw

@portfolio.command()
@click.option('--id', 'optid', help=u'ids of portfolio to update')
@click.option('--type', 'opttype', default='8,9', help=u'which type to run')
@click.option('--risk', 'optrisk', default='1,2,3,4,5,6,7,8,9,10', help=u'which risk to update')
@click.option('--fee', 'optfee', default='9,8', help=u'fee type(8:with fee; 9:without fee')
@click.option('--debug/--no-debug', 'optdebug', default=False, help=u'debug mode')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.option('--end-date', 'optenddate', default=None, help=u'calc end date for nav')
@click.pass_context
def nav(ctx, optid, opttype, optlist, optrisk, optfee, optdebug, optenddate):
    ''' calc pool nav and inc
    '''
    if optid is not None:
        portfolios = [s.strip() for s in optid.split(',')]
    else:
        if 'portfolio' in ctx.obj:
            portfolios = [str(ctx.obj['portfolio'])]
        else:
            portfolios = None

    fees = [int(s.strip()) for s in optfee.split(',')]
    risks = [int(s.strip()) for s in optrisk.split(',')]

    xtypes = [s.strip() for s in opttype.split(',')]

    df_portfolio = asset_ra_portfolio.load(portfolios, xtypes)

    if optlist:
        df_portfolio['ra_name'] = df_portfolio['ra_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_portfolio, headers='keys', tablefmt='psql')
        return 0

    for fee in fees:
        for _, portfolio in df_portfolio.iterrows():
            nav_update_alloc(portfolio, risks, fee, optdebug, optenddate)

def nav_update_alloc(portfolio, risks, fee, debug, enddate):
    df_alloc = asset_ra_portfolio_alloc.where_portfolio_id(portfolio['globalid'])
    df_alloc = df_alloc.loc[(df_alloc['ra_risk'] * 10).astype(int).isin(risks)]

    feestr = 'FEE' if fee == 8 else 'NOF'
    
    with click.progressbar(
            df_alloc.iterrows(), length=len(df_alloc.index),
            label=('update nav %-9s (%s)' % (portfolio['globalid'], feestr)).ljust(30),
            item_show_func=lambda x: str(x[1]['globalid']) if x else None) as bar:
        for _, alloc in bar:
            nav_update(alloc, fee, debug, enddate)
    
def nav_update(alloc, fee, debug, enddate):
    alloc_id = alloc['globalid']

    # 加载仓位信息
    df_pos = asset_ra_portfolio_pos.load_fund_pos(alloc_id)
    if df_pos.empty:
        click.echo(click.style("\nswarning: empty df_pos for alloc %s, skiped!" % (alloc_id), fg='yellow'))
        return

    if enddate is not None:
        max_date = pd.to_datetime(enddate)
    else:
        max_date = (datetime.now() - timedelta(days=1)) # yesterday

    # 计算复合资产净值
    if fee == 8:
        xtype = 8
        df_pos = df_pos.loc[df_pos.index.get_level_values(0) >= '2012-07-27']
        tn = TradeNav.TradeNav(debug=debug)
        tn.calc(df_pos, 1)
        sr_nav_portfolio = pd.Series(tn.nav)
        sr_contrib = pd.concat(tn.contrib)
    else:
        xtype = 9
        #print "df_pos", df_pos
        #print "max_date", max_date
        sr_nav_portfolio = DFUtil.portfolio_nav2(df_pos, end_date=max_date)
        sr_contrib = pd.Series()

    df_result = sr_nav_portfolio.to_frame('ra_nav')
    df_result.index.name = 'ra_date'
    df_result['ra_type'] = xtype
    df_result['ra_inc'] = df_result['ra_nav'].pct_change().fillna(0.0)
    df_result['ra_portfolio_id'] = alloc['globalid']
    df_result = df_result.reset_index().set_index(['ra_portfolio_id', 'ra_type', 'ra_date'])

    asset_ra_portfolio_nav.save(alloc_id, xtype, df_result)

    if not sr_contrib.empty:
        df_contrib = sr_contrib.to_frame('ra_return_value')
        df_contrib.index.names=[u'ra_date', u'ra_return_type', u'ra_fund_id']
        df_contrib['ra_type'] = xtype
        df_contrib['ra_portfolio_id'] = alloc_id
        df_contrib = df_contrib.reset_index().set_index(['ra_portfolio_id', 'ra_type', 'ra_date', 'ra_fund_id', 'ra_return_type'])

        asset_ra_portfolio_contrib.save(alloc_id, xtype, df_contrib)

@portfolio.command()
@click.option('--id', 'optid', help=u'ids of portfolio to update')
@click.option('--type', 'opttype', default='8,9', help=u'which type to run')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.pass_context
def turnover(ctx, optid, opttype, optlist):
    ''' calc pool turnover and inc
    '''
    if optid is not None:
        portfolios = [s.strip() for s in optid.split(',')]
    else:
        if 'portfolio' in ctx.obj:
            portfolios = [str(ctx.obj['portfolio'])]
        else:
            portfolios = None
            
    xtypes = [s.strip() for s in opttype.split(',')]

    df_portfolio = asset_ra_portfolio.load(portfolios, xtypes)

    if optlist:

        df_portfolio['ra_name'] = df_portfolio['ra_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_portfolio, headers='keys', tablefmt='psql')
        return 0
    
    data = []
    for _, portfolio in df_portfolio.iterrows():
        turnover_update_alloc(portfolio)

def turnover_update_alloc(portfolio):
    df_alloc = asset_ra_portfolio_alloc.where_portfolio_id(portfolio['globalid'])
    
    with click.progressbar(
            df_alloc.iterrows(), length=len(df_alloc.index),
            label=('turnover %-11s' % (portfolio['globalid'])).ljust(30),
            item_show_func=lambda x:  str(x[1]['globalid']) if x else None) as bar:
        for _, alloc in bar:
            turnover_update(alloc)

            
def turnover_update(portfolio):
    portfolio_id = portfolio['globalid']
    # 加载仓位信息
    df = asset_ra_portfolio_pos.load_fund_pos(portfolio_id)
    df = df.unstack()


    # 计算宽口换手率
    sr_turnover = DFUtil.calc_turnover(df)

    criteria_id = 6
    df_result = sr_turnover.to_frame('ra_value')
    df_result['ra_portfolio_id'] = portfolio_id
    df_result['ra_criteria_id'] = criteria_id
    df_result = df_result.reset_index().set_index(['ra_portfolio_id', 'ra_criteria_id', 'ra_date'])

    asset_ra_portfolio_criteria.save(portfolio_id, criteria_id,  df_result)

    total_turnover = sr_turnover.sum()

    return total_turnover

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
# @click.option('-m', '--msg')  
# @click.option('--dry-run', is_flag=True, help=u'pretend to run')
# @click.option('--name', prompt='Your name', help='The person to greet.')
@click.pass_context
def ncat(ctx, datadir):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    GeneralizationPosition.portfolio_category()

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
# @click.option('-m', '--msg')  
# @click.option('--dry-run', is_flag=True, help=u'pretend to run')
# @click.option('--name', prompt='Your name', help='The person to greet.')
@click.pass_context
def nsimple(ctx, datadir):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    GeneralizationPosition.portfolio_simple()

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.pass_context
def detail(ctx, datadir):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    GeneralizationPosition.portfolio_detail()

@portfolio.command()  
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--output', '-o', default=None, help=u'file used to store final result')
@click.pass_context
def trade(ctx, datadir, output):
    '''generate final portfolio with optimized strategy (cost consider in).  
    '''
    Const.datadir = datadir
    if output is None:
        output = datapath('position-z.csv')
    with (open(output, 'w') if output != '-' else os.fdopen(os.dup(sys.stdout.fileno()), 'w')) as out:
        GeneralizationPosition.portfolio_trade(out)

@portfolio.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--input', '-i', 'optinput', type=click.Path(exists=True), help=u'portfolio position file as input') 
@click.option('--output', '-o', 'optoutput', type=click.Path(), help=u'file position file to output') 
@click.pass_context
def stockavg(ctx, datadir, optinput, optoutput):
    '''generate final portfolio using simple average strategy (no cost)
    '''
    Const.datadir = datadir
    #
    # 生成配置数据
    #
    output = None
    if optinput is None:
        if os.path.isfile(datapath('riskmgr_position.csv')):
            optinput = datapath('riskmgr_position.csv')
            output = datapath('position-r.csv')
        elif os.path.isfile(datapath('portfolio_position.csv')):
            optinput = datapath('portfolio_position.csv')
            output = datapath('position-v.csv')
        else:
            click.echo(click.style("error: mising position file!", fg="yellow"))
            return -1
        
    if optoutput is None:
        if output is None:
            optoutput = datapath('position-v.csv')
        else:
            optoutput = output
                
        
    print "convert portfilio position  %s to final position %s" % (optinput, optoutput)
    GeneralizationPosition.portfolio_avg_simple(optinput, optoutput)

@portfolio.command()
@click.option('--inst', 'optInst', help=u'portfolio id to calc turnover')
@click.option('--alloc', 'optAlloc', help=u'risk of portfolio to calc turnover')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.option('--start-date', 'startdate', default='2010-01-08', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.pass_context
def turnover_o(ctx, optInst, optAlloc, startdate, enddate, optlist):
    '''run constant risk model
    '''    
    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")        
    
    db_asset = create_engine(config.db_asset_uri)
    # db_asset.echo = True
    # db_base = create_engine(config.db_base_uri)
    db = {'asset':db_asset}

    df = load_portfolio_by_id(db['asset'], optInst, optAlloc)

    df_result = calc_turnover(df)

    total_turnover = df_result.sum()

    df_result.reset_index(inplace=True)
    df_result['turnover'] = df_result['turnover'].map(lambda x: "%6.2f%%" % (x * 100))

    print tabulate(df_result, headers='keys', tablefmt='psql', stralign=u'right')

    print "total turnover: %.2f%%" % (total_turnover * 100)
    
    # if optlist:
    #     #print df_pool
    #     #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
    #     df_pool['ra_name'] = df_pool['ra_name'].map(lambda e: e.decode('utf-8'))
    #     print tabulate(df_pool, headers='keys', tablefmt='psql')
    #     return 0
    
    # for _, pool in df_pool.iterrows():
    #     stock_update(db, pool, optlimit, optcalc)

def calc_turnover(df):

    df.loc[(df['ai_category'] >= 10) & (df['ai_category'] < 30), 'ai_category'] = (df['ai_category'] // 10)
    df.set_index(['ai_transfer_date', 'ai_category', 'ai_fund_code'], inplace=True)
    #df2 = df.groupby(level=(0,1)).agg({'ai_inst_type':'first', 'ai_alloc_id':'first', 'ai_fund_ratio':'sum'})
    df2 = df[['ai_fund_ratio']].groupby(level=(0,1)).sum()
    df2 = df2.unstack().fillna(0.0)
    df2.columns = df2.columns.droplevel(0)

    df_result = df2.rolling(window=2, min_periods=1).apply(lambda x: x[1] - x[0] if len(x) > 1 else x[0])

    df_result = df_result.abs().sum(axis=1).to_frame('turnover')

    return df_result


def load_portfolio_by_id(db, inst, alloc):
    metadata = MetaData(bind=db)
    t = Table('allocation_instance_position_detail', metadata, autoload=True)

    columns = [
        t.c.ai_inst_type,
        t.c.ai_alloc_id,
        t.c.ai_transfer_date,
        t.c.ai_category,
        t.c.ai_fund_code,
        t.c.ai_fund_ratio,
    ]
    s = select(columns, and_(t.c.ai_inst_id == inst, t.c.ai_alloc_id == alloc)) \
        .order_by(t.c.ai_transfer_date, t.c.ai_category)    
    df = pd.read_sql(s, db, parse_dates=['ai_transfer_date'])

    return df

def load_portfolio_category_by_id(db, inst, alloc):
    metadata = MetaData(bind=db)
    t = Table('allocation_instance_position_detail', metadata, autoload=True)

    columns = [
        t.c.ai_alloc_id,
        t.c.ai_transfer_date,
        t.c.ai_category,
        func.sum(t.c.ai_fund_ratio).label('ratio')
    ]
    s = select(columns, and_(t.c.ai_inst_id == inst, t.c.ai_alloc_id == alloc)) \
        .group_by(t.c.ai_transfer_date, t.c.ai_category) \
        .order_by(t.c.ai_transfer_date, t.c.ai_category)    
    df = pd.read_sql(s, db, parse_dates=['ai_transfer_date'])

    return df


@portfolio.command()
@click.option('--from', 'optfrom', help=u'portfolio id to convert from')
@click.option('--to', 'optto', help=u'portfolio id to convert to')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.pass_context
def convert1(ctx, optfrom, optto, optlist):
    '''convert bond and money to bank
    '''    

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('allocation_instance_position_detail', metadata, autoload=True)
    t2 = Table('allocation_instance_position', metadata, autoload=True)

    columns = [
        t1.c.ai_inst_type,
        t1.c.ai_alloc_id,
        t1.c.ai_transfer_date,
        t1.c.ai_category,
        t1.c.ai_fund_id,
        t1.c.ai_fund_code,
        t1.c.ai_fund_ratio,
    ]

    s = select(columns).where(t1.c.ai_inst_id == optfrom)
        
    df = pd.read_sql(s, db, index_col = ['ai_alloc_id', 'ai_transfer_date'], parse_dates=['ai_transfer_date'])

    mask = (df['ai_category'] >= 20) & (df['ai_category'] < 30)
    df.loc[mask, 'ai_fund_id'] =  33056942
    df.loc[mask, 'ai_fund_code'] = '999002'
    #df.loc[maks, 'ai_category'] = 51

    mask = (df['ai_category'] >= 30) & (df['ai_category'] < 40)
    df.loc[mask, 'ai_fund_id'] =  33056941
    df.loc[mask, 'ai_fund_code'] = '999001'
    #df.loc[maks, 'ai_category'] = 52

    df.set_index(['ai_category', 'ai_fund_id'], append=True, inplace=True)
    df_result = df.groupby(level=(0, 1, 2, 3)).agg({'ai_inst_type':'first', 'ai_fund_code':'first', 'ai_fund_ratio':'sum'})

    t1.delete(t1.c.ai_inst_id == optto).execute()

    df_result['ai_inst_id'] = optto
    df_result['updated_at'] = df_result['created_at'] = datetime.now()

    df_result = df_result.reset_index().set_index(['ai_inst_id', 'ai_alloc_id', 'ai_transfer_date', 'ai_category', 'ai_fund_id'])
    
    df_result.to_sql(t1.name, db, index=True, if_exists='append', flavor='mysql', chunksize=500)


    df_result = df_result.reset_index().set_index(['ai_inst_id', 'ai_alloc_id', 'ai_transfer_date', 'ai_fund_id'])
    df_result['ai_fund_type'] = df_result['ai_category'].floordiv(10)
    df_result =  df_result.drop('ai_category', axis = 1)
    
    df_result = df_result.groupby(level=(0, 1, 2, 3)).agg({
        'ai_inst_type':'first', 'ai_fund_code':'first', 'ai_fund_type':'first', 'ai_fund_ratio':'sum', 'updated_at':'first', 'created_at':'first'
    })
    #print df_result.head()

    t2.delete(t2.c.ai_inst_id == optto).execute()
    df_result.to_sql(t2.name, db, index=True, if_exists='append', flavor='mysql', chunksize=500)
    

@portfolio.command()
@click.option('--src', 'optsrc', help=u'src id of portfolio to copy from')
@click.option('--dst', 'optdst', help=u'dst id of portfolio to copy to')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list instance to update')
@click.pass_context
def copy(ctx, optsrc, optdst, optlist):
    ''' create new portfolio by copying  existed one
    '''
    if optsrc is not None:
        portfolios = [optsrc]
    else:
        portfolios = None

    df_portfolio = asset_ra_portfolio.load(portfolios)

    if optlist:

        df_portfolio['ra_name'] = df_portfolio['ra_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_portfolio, headers='keys', tablefmt='psql')
        return 0

    if optsrc  is None or optdst is None:
        click.echo(click.style("\n both --src-id  and --dst-id is required to perform copy\n", fg='red'))
        return 0

    #
    # copy ra_portfolio
    #
    df_portfolio['globalid'] = optdst
    df_portfolio.set_index(['globalid'], inplace=True)
    asset_ra_portfolio.save(optdst, df_portfolio)

    #
    # copy ra_portfolio_alloc
    #
    df_portfolio_alloc = asset_ra_portfolio_alloc.where_portfolio_id(optsrc)
    # df_portfolio_alloc.reset_index(inplace=True)

    df_portfolio_alloc['ra_portfolio_id'] = optdst
    df_portfolio_alloc['old'] = df_portfolio_alloc['globalid']
    sr_tmp = df_portfolio_alloc['ra_portfolio_id'].str[:len(optdst) - 1]
    df_portfolio_alloc['globalid'] = sr_tmp.str.cat((df_portfolio_alloc['ra_risk'] * 10 % 10).astype(int).astype(str))

    df_xtab = df_portfolio_alloc[['globalid', 'old']].copy()

    df_portfolio_alloc.drop(['old'], axis=1, inplace=True)
    
    df_portfolio_alloc.set_index(['globalid'], inplace=True)
    asset_ra_portfolio_alloc.save(optdst, df_portfolio_alloc)

    #
    # copy ra_portfolio_argv
    #
    df_portfolio_argv = asset_ra_portfolio_argv.load(df_xtab['old'])
    df_portfolio_argv.reset_index(inplace=True)

    df_portfolio_argv = df_portfolio_argv.merge(df_xtab, left_on='ra_portfolio_id', right_on = 'old')
    df_portfolio_argv['ra_portfolio_id'] = df_portfolio_argv['globalid']
    df_portfolio_argv.drop(['globalid', 'old'], inplace=True, axis=1)
    df_portfolio_argv = df_portfolio_argv.set_index(['ra_portfolio_id', 'ra_key'])

    asset_ra_portfolio_argv.save(df_xtab['globalid'], df_portfolio_argv)

    #
    # copy ra_portfolio_asset
    #
    df_portfolio_asset = asset_ra_portfolio_asset.load(df_xtab['old'])
    # df_portfolio_asset.reset_index(inplace=True)

    df_portfolio_asset = df_portfolio_asset.merge(df_xtab, left_on='ra_portfolio_id', right_on = 'old')

    df_portfolio_asset['ra_portfolio_id'] = df_portfolio_asset['globalid']
    df_portfolio_asset.drop(['globalid', 'old'], inplace=True, axis=1)
    df_portfolio_asset = df_portfolio_asset.set_index(['ra_portfolio_id', 'ra_asset_id'])

    asset_ra_portfolio_asset.save(df_xtab['globalid'], df_portfolio_asset)

    
