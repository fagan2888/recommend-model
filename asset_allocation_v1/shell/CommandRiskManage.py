#coding=utf8


import logging
import string
import json
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import os
import util_numpy as npu
import DBData
import time
import Const
import RiskManagement
import DFUtil

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from tabulate import tabulate
from sqlalchemy import MetaData, Table, select, func
from db import database, asset_rm_risk_mgr, asset_rm_riskmgr, asset_tc_timing_signal, asset_rm_riskmgr_signal, asset_rm_riskmgr_nav

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)  
@click.option('--id', 'optid', help=u'reshape id')
@click.option('--online/--no-online', 'optonline', default=False, help=u'include online instance')
@click.pass_context
def riskmgr(ctx, optid, optonline):
    '''risk management group
    '''
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        ctx.invoke(signal, optid=optid, optonline=optonline)
        ctx.invoke(nav, optid=optid)
    else:
        # click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        pass

@riskmgr.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2012-07-15', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.option('--label-asset/--no-label-asset', default=True)
@click.option('--reshape/--no-reshape', default=True)
@click.option('--markowitz/--no-markowitz', default=True)
@click.pass_context
def test(ctx, datadir, startdate, enddate, label_asset, reshape, markowitz):
    '''run risk management using simple strategy
    '''
    Const.datadir = datadir

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")        

    df_nav = pd.read_csv(datapath('../csvdata/000300_nav_2012.csv'),  index_col=['date'], parse_dates=['date'])
    #df_nav = pd.read_csv(datapath('dd.csv'),  index_col=['td_date'], parse_dates=['td_date'])

    #df_pos = pd.read_csv(datapath('port_weight.csv'),  index_col=['date'], parse_dates=['date'])

    df_timing = pd.read_csv(datapath('hs_gftd.csv'),  index_col=['date'], parse_dates=['date'], usecols=['date', 'trade_types'])
    #df_timing = pd.read_csv(datapath('../csvdata/000300_gftd_result.csv'),  index_col=['date'], parse_dates=['date'], usecols=['date', 'trade_types'])
    df_timing = df_timing.rename(columns={'trade_types':'sh000300'})

    df = pd.DataFrame({
        'nav': df_nav.iloc[:, 0],
        'timing': df_timing['sh000300'].reindex(df_nav.index, method='pad')
    })

    risk_mgr = RiskManagement.RiskManagement()
    df_result = risk_mgr.perform('sh000300', df)

    df_result.to_csv(datapath('riskmgr_result.csv'))

@riskmgr.command()
@click.option('--inst', 'optinst', type=int, help=u'risk mgr id')
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--start-date', 'startdate', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.pass_context
def simple(ctx, datadir, optinst, startdate, enddate):
    '''run risk management using simple strategy
    '''
    Const.datadir = datadir

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")
        
    if optinst is None:
        optinst = 2016120600

    make_rm_risk_mgr_if_not_exist(optinst)

    df_nav = pd.read_csv(datapath('equalriskasset.csv'),  index_col=['date'], parse_dates=['date'])
    if not startdate:
        startdate = df_nav.index.min().strftime("%Y-%m-%d")

    timing_ids = ['49101', '49201', '49301', '49401']
    df_timing = database.asset_tc_timing_signal_load(
        timing_ids, begin_date=startdate, end_date=enddate)

    tasks = {
        'largecap'       : [49101, 11],
        'smallcap'       : [49101, 12],
        'rise'           : [49101, 13],
        'oscillation'    : [49101, 14],
        'decline'        : [49101, 15],
        'growth'         : [49101, 16],
        'value'          : [49101, 17],
        'SP500.SPI'      : [49201, 41],
        'GLNC'           : [49301, 42],
        'HSCI.HI'        : [49401, 43],
        'convertiblebond': None,
        'creditbond'     : None,
        'money'          : None,
        'ratebond'       : None,
    }

    data = {}
    risk_mgr = RiskManagement.RiskManagement()
    for asset in df_nav.columns:
        #
        # 计算风控仓位
        #
        if tasks[asset] is None:
            continue

        timing_id, category = tasks[asset]

        df = pd.DataFrame({
            'nav': df_nav[asset],
            'timing': df_timing[timing_id].reindex(df_nav.index, method='pad')
        })
        
        df_new = risk_mgr.perform(asset, df)
        
        df_new['rm_risk_mgr_id'] = optinst
        df_new['rm_category'] = category
        df_new = df_new.reset_index().set_index(['rm_risk_mgr_id', 'rm_category', 'rm_date'])
        if not df_new.empty:
            df_new = df_new.applymap("{:.0f}".format)
        
        #
        # 保存风控仓位到数据库
        #
        db = database.connection('asset')
        t2 = Table('rm_risk_mgr_signal', MetaData(bind=db), autoload=True)
        columns2 = [
            t2.c.rm_risk_mgr_id,
            t2.c.rm_category,
            t2.c.rm_date,
            t2.c.rm_action,
            t2.c.rm_pos,
        ]
        s = select(columns2, (t2.c.rm_risk_mgr_id == optinst) & (t2.c.rm_category == category))
        df_old = pd.read_sql(s, db, index_col=['rm_risk_mgr_id', 'rm_category', 'rm_date'], parse_dates=['rm_date'])
        if not df_old.empty:
            df_old = df_old.applymap("{:.0f}".format)

        # 更新数据库
        # print df_new.head()
        # print df_old.head()
        database.batch(db, t2, df_new, df_old, timestamp=False)

    #
    # 合并 markowitz 仓位 与 风控 结果
    #
    df_pos_markowitz = pd.read_csv(datapath('portfolio_position.csv'), index_col=['date'], parse_dates=['date'])

    df_pos_riskmgr = database.asset_rm_risk_mgr_signal_load(optinst)

    for column in df_pos_riskmgr.columns:
        category = DFUtil.categories_name(column)
        # if column < 20:
        #     rmc = 11
        # else:
        #     rmc = column
        rmc = column
        # print "use column %d for category %s" % (rmc, category)
                
        if category in df_pos_markowitz:
            df_pos_tmp = df_pos_riskmgr[rmc].reindex(df_pos_markowitz.index)
            df_pos_markowitz[category] = df_pos_markowitz[category] * df_pos_tmp

    df_result = df_pos_markowitz.reset_index().set_index(['risk', 'date'])

    #
    # 调整货币的比例, 总和达到1
    #
    df_result['money'] = (1 - (df_result.sum(axis=1) - df_result['money']))
    
    df_result.to_csv(datapath('riskmgr_position.csv'))

def make_rm_risk_mgr_if_not_exist(id_):
    db = database.connection('asset')
    t2 = Table('rm_risk_mgr', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.globalid,
        t2.c.rm_inst_id,
    ]
    s = select(columns2, (t2.c.globalid == id_))
    df = pd.read_sql(s, db, index_col=['globalid'])
    if not df.empty:
        return True
    #
    # 导入数据
    #
    row = {
        'globalid': id_, 'rm_inst_id': id_, 'created_at': func.now(), 'updated_at': func.now()
    }
    t2.insert(row).execute()

    return True

@riskmgr.command(name='import')
@click.option('--id', 'optid', type=int, help=u'specify markowitz id')
@click.option('--name', 'optname', type=int, help=u'specify markowitz name')
@click.option('--type', 'opttype', type=click.Choice(['1', '9']), default='1', help=u'online type(1:expriment; 9:online)')
@click.option('--replace/--no-replace', 'optreplace', default=False, help=u'replace pool if exists')
@click.argument('csv', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=False), required=True)
@click.pass_context
def import_command(ctx, csv, optid, optname, opttype, optreplace):
    '''
    import risk management position from csv file
    '''

    #
    # 处理id参数
    #
    if optid is not None:
        #
        # 检查id是否存在
        #
        df_existed = asset_rm_risk_mgr.load([str(optid)])
        if not df_existed.empty:
            s = 'riskmgr instance [%d] existed' % optid
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
        prefix = '60' + today.strftime("%m%d");
        if opttype == '9':
            between_min, between_max = ('%s90' % (prefix), '%s99' % (prefix))
        else:
            between_min, between_max = ('%s00' % (prefix), '%s89' % (prefix))

        max_id = asset_rm_risk_mgr.max_id_between(between_min, between_max)
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
    rm_risk_mgr = Table('rm_risk_mgr', metadata, autoload=True)
    rm_risk_mgr_pos = Table('rm_risk_mgr_pos', metadata, autoload=True)
    # rm_risk_mgr_nav = Table('rm_risk_mgr_nav', metadata, autoload=True)

    #
    # 处理替换
    #
    if optreplace:
        rm_risk_mgr.delete(rm_risk_mgr.c.globalid == optid).execute()
        rm_risk_mgr_pos.delete(rm_risk_mgr_pos.c.rm_risk_mgr_id == optid).execute()
        # rm_risk_mgr_nav.delete(rm_risk_mgr_nav.c.rm_risk_mgr_id == optid).execute()

    now = datetime.now()
    #
    # 导入数据
    #
    row = {
        'globalid': optid, 'rm_type':opttype, 'rm_name': optname,
        'rm_pool': '', 'rm_reshape': '', 'rm_markowitz': '', 'created_at': func.now(), 'updated_at': func.now()
    }
    rm_risk_mgr.insert(row).execute()

    df = pd.read_csv(csv, parse_dates=['date'])
    df['risk'] = (df['risk'] * 10).astype(int)
    renames = dict(
        {'date':'rm_date', 'risk':'rm_alloc_id'}.items() + DFUtil.categories_types(as_int=True).items()
    )
    df = df.rename(columns=renames)
    df['rm_risk_mgr_id'] = optid

    df.set_index(['rm_risk_mgr_id', 'rm_alloc_id', 'rm_date'], inplace=True)

    # 四舍五入到万分位
    df = df.round(4)
    # 过滤掉过小的份额
    df[df.abs() < 0.0009999] = 0
    # 补足缺失
    df = df.apply(npu.np_pad_to, raw=True, axis=1)
    # 过滤掉相同
    df = df.groupby(level=(0,1), group_keys=False).apply(DFUtil.filter_same_with_last)

    df.columns.name='rm_asset'
    df_tosave = df.stack().to_frame('rm_ratio')
    df_tosave = df_tosave.loc[df_tosave['rm_ratio'] > 0, ['rm_ratio']]
    if not df_tosave.empty:
        database.number_format(df_tosave, columns=['rm_ratio'], precision=4)
    
    df_tosave['updated_at'] = df_tosave['created_at'] = now

    df_tosave.to_sql(rm_risk_mgr_pos.name, db, index=True, if_exists='append', chunksize=500)

    if len(df_tosave.index) > 1:
        logger.info("insert %s (%5d) : %s " % (rm_risk_mgr_pos.name, len(df_tosave.index), df_tosave.index[0]))

    click.echo(click.style("import complement! instance id [%s]" % (optid), fg='green'))
    
    return 0

@riskmgr.command()
@click.option('--id', 'optid', help=u'risk mgr id')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.option('--online/--no-online', 'optonline', default=False, help=u'include online instance')
@click.pass_context
def signal(ctx, optid, optlist, optonline):
    '''run risk management using simple strategy
    '''

    if optid is not None:
        ids = [s.strip() for s in optid.split(',')]
    else:
        ids = None

    xtypes = None
    if optonline == False:
        xtypes = [1]

    df_riskmgr = asset_rm_riskmgr.load(ids, xtypes)

    if optlist:

        df_riskmgr['rm_name'] = df_riskmgr['rm_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_riskmgr, headers='keys', tablefmt='psql')
        return 0
    
    # with click.progressbar(length=len(df_riskmgr), label='update riskmgr signal') as bar:
    #     for _, riskmgr in df_riskmgr.iterrows():
    #         bar.update(1)
    #         signal_update(riskmgr)
    for _, riskmgr in df_riskmgr.iterrows():
        signal_update(riskmgr)

def signal_update(riskmgr):
    riskmgr_id = riskmgr['globalid']
    
    # 加载择时信号
    sr_timing = asset_tc_timing_signal.load_series(riskmgr['rm_timing_id'])
    # print sr_timing.head()
    if sr_timing.empty:
        click.echo(click.style("\nempty timing signal (%d, %d)\n" % (riskmgr_id, riskmgr['rm_timing_id']), fg="red"))
    
    # 加载资产收益率
    # min_date = df_position.index.min()
    # max_date = (datetime.now() - timedelta(days=1)) # yesterday

    # sr_nav = asset_ra_pool_nav.load_series(
    #     riskmgr['rm_pool'], riskmgr['rm_asset'], riskmgr['rm_type'])
    if riskmgr['rm_start_date'] != '0000-00-00':
        sdate = riskmgr['rm_start_date']
    else:
        sdate = None
    sr_nav = database.load_nav_series(riskmgr['rm_asset_id'], sdate)
    
    # df_inc = df_nav.pct_change().fillna(0.0).to_frame(riskmgr_id)
    df = pd.DataFrame({'nav': sr_nav, 'timing': sr_timing})

    risk_mgr = RiskManagement.RiskManagement()
    df_result = risk_mgr.perform(riskmgr_id, df)
    # df_result.drop(['nav', 'timing'], axis=1, inplace=True)
    df_result = DFUtil.filter_same_with_last(df_result)

    # df_result = df_nav_portfolio[['portfolio']].rename(columns={'portfolio':'rm_nav'}).copy()
    df_result.index.name = 'rm_date'
    df_result['rm_riskmgr_id'] = riskmgr_id
    df_tosave = df_result.reset_index().set_index(['rm_riskmgr_id', 'rm_date'])

    asset_rm_riskmgr_signal.save(riskmgr_id, df_tosave)

@riskmgr.command()
@click.option('--id', 'optid', help=u'ids of fund pool to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.pass_context
def nav(ctx, optid, optlist):
    ''' calc riskmgr nav and inc
    '''
    if optid is not None:
        ids = [s.strip() for s in optid.split(',')]
    else:
        ids = None

    df_riskmgr = asset_rm_riskmgr.load(ids)

    if optlist:

        df_riskmgr['rm_name'] = df_riskmgr['rm_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_riskmgr, headers='keys', tablefmt='psql')
        return 0
    
    with click.progressbar(length=len(df_riskmgr), label='update nav') as bar:
        for _, riskmgr in df_riskmgr.iterrows():
            bar.update(1)
            nav_update(riskmgr)

def nav_update(riskmgr):
    riskmgr_id = riskmgr['globalid']
    # 加载择时信号
    sr_position = asset_rm_riskmgr_signal.load_series(riskmgr_id)
    if sr_position.empty:
        return
    df_position = sr_position.to_frame(riskmgr_id)

    # 加载基金收益率
    min_date = df_position.index.min()
    #max_date = df_position.index.max()
    max_date = (datetime.now() - timedelta(days=1)) # yesterday


    sr_nav = database.load_nav_series(
        riskmgr['rm_asset_id'], begin_date=min_date, end_date=max_date);
    df_inc = sr_nav.pct_change().fillna(0.0).to_frame(riskmgr_id)

    # 计算复合资产净值
    df_nav_portfolio = DFUtil.portfolio_nav(df_inc, df_position, result_col='portfolio')

    df_result = df_nav_portfolio[['portfolio']].rename(columns={'portfolio':'rm_nav'}).copy()
    df_result.index.name = 'rm_date'
    df_result['rm_inc'] = df_result['rm_nav'].pct_change().fillna(0.0)
    df_result['rm_riskmgr_id'] = riskmgr_id
    df_result = df_result.reset_index().set_index(['rm_riskmgr_id', 'rm_date'])
    
    asset_rm_riskmgr_nav.save(riskmgr_id, df_result)
