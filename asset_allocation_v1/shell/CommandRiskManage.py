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
import RiskManagement
import database
import DFUtil

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table,select

import traceback, code

@click.group()  
@click.pass_context
def riskmgr(ctx):
    '''risk management group
    '''
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
        t2 = Table('rm_risk_mgr_pos', MetaData(bind=db), autoload=True)
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

    df_pos_riskmgr = database.asset_rm_risk_mgr_pos_load(optinst)

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

