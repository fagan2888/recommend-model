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
import Const
import database
import DFUtil
from TimingGFTD import TimingGFTD

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from tabulate import tabulate


import traceback, code

@click.group()  
@click.pass_context
def timing(ctx):
    '''timing group
    '''
    pass



@timing.command()
@click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
@click.option('--start-date', 'startdate', default='2012-07-15', help=u'start date to calc')
@click.option('--end-date', 'enddate', help=u'end date to calc')
@click.pass_context
def test(ctx, datadir, startdate, enddate):
    '''run risk management using simple strategy
    '''
    Const.datadir = datadir

    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1)); 
        enddate = yesterday.strftime("%Y-%m-%d")        

    df_nav = pd.read_csv(datapath('000300_gftd_result.csv'),  index_col=['date'], parse_dates=['date'], usecols=['date', 'open', 'high', 'low', 'close'])
    # print df_nav.head()
    df_nav.rename(columns={'open':'tc_open', 'high':'tc_high', 'low':'tc_low', 'close':'tc_close'}, inplace=True)
    df_nav.index.name='tc_date'

    # df_timing = pd.read_csv(datapath('hs_gftd.csv'),  index_col=['date'], parse_dates=['date'], usecols=['date', 'trade_types'])
    # #df_timing = pd.read_csv(datapath('../csvdata/000300_gftd_result.csv'),  index_col=['date'], parse_dates=['date'], usecols=['date', 'trade_types'])
    # df_timing = df_timing.rename(columns={'trade_types':'sh000300'})

    # df = pd.DataFrame({
    #     'nav': df_nav.iloc[:, 0],
    #     'timing': df_timing['sh000300'].reindex(df_nav.index, method='pad')
    # })

    #df_nav = df_nav.loc[:'2010-03-18', :]

    # risk_mgr = RiskManagement.RiskManagement()
    df_new = TimingGFTD().timing(df_nav)
    df_new['tc_timing_id'] = 41101
    df_new = df_new.reset_index().set_index(['tc_timing_id', 'tc_date'])

    # print df_new[df_new['tc_stop'].isnull()].head()
    num_signal = df_new['tc_signal'].rolling(2, 1).apply(lambda x: 1 if x[-1] != x[0] else 0).sum()
    
    formaters = ['tc_close', 'tc_open', 'tc_high', 'tc_low', 'tc_recording_high', 'tc_recording_low', 'tc_stop']

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
        t2.c.tc_close,
        t2.c.tc_open,
        t2.c.tc_high,
        t2.c.tc_low,
        t2.c.tc_ud,
        t2.c.tc_ud_flip,
        t2.c.tc_ud_acc,
        t2.c.tc_buy_start,
        t2.c.tc_buy_kstick,
        t2.c.tc_buy_count,
        t2.c.tc_buy_signal,
        t2.c.tc_sell_start,
        t2.c.tc_sell_kstick,
        t2.c.tc_sell_count,
        t2.c.tc_sell_signal,
        t2.c.tc_action,
        t2.c.tc_recording_high,
        t2.c.tc_recording_low,
        t2.c.tc_signal,
        t2.c.tc_stop,
    ]
    s = select(columns2, (t2.c.tc_timing_id == 41101))
    df_old = pd.read_sql(s, db, index_col=['tc_timing_id', 'tc_date'], parse_dates=['tc_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, columns=formaters, precision=4)

    # 更新数据库
    # print df_new.head()
    # print df_old.head()
    database.batch(db, t2, df_new, df_old, timestamp=False)
    print "total signal: %d, %.2f/year" % (num_signal, num_signal * 250/len(df_new))
    

@timing.command()
@click.option('--id', 'optid', help=u'ids of fund pool to update')
@click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
@click.pass_context
def nav(ctx, optid, optlist):
    ''' calc pool nav and inc
    '''
    if optid is not None:
        timings = [s.strip() for s in optid.split(',')]
    else:
        timings = None

    df_timing = database.asset_tc_timing_load(timings)

    if optlist:

        df_timing['tc_name'] = df_timing['tc_name'].map(lambda e: e.decode('utf-8'))
        print tabulate(df_timing, headers='keys', tablefmt='psql')
        return 0
    
    with click.progressbar(length=len(df_timing), label='update nav') as bar:
        for _, timing in df_timing.iterrows():
            bar.update(1)
            nav_update(timing)

def nav_update(timing):
    timing_id = timing['globalid']
    # 加载择时信号
    df_position = database.asset_tc_timing_scratch_load_signal(timing_id)
    # 构建仓位
    df_position.loc[df_position[timing_id] < 1, timing_id] = 0
    
    # 加载基金收益率
    min_date = df_position.index.min()
    #max_date = df_position.index.max()
    max_date = (datetime.now() - timedelta(days=1)) # yesterday


    df_nav = database.base_ra_index_nav_load_series(
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

    # df_result.to_csv(datapath('riskmgr_result.csv'))

# @timing.command()
# @click.option('--inst', 'optinst', type=int, help=u'risk mgr id')
# @click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
# @click.option('--start-date', 'startdate', help=u'start date to calc')
# @click.option('--end-date', 'enddate', help=u'end date to calc')
# @click.pass_context
# def simple(ctx, datadir, optinst, startdate, enddate):
#     '''run risk management using simple strategy
#     '''
#     Const.datadir = datadir

#     if not enddate:
#         yesterday = (datetime.now() - timedelta(days=1)); 
#         enddate = yesterday.strftime("%Y-%m-%d")
        
#     if optinst is None:
#         optinst = 2016120600

#     make_rm_risk_mgr_if_not_exist(optinst)

#     df_nav = pd.read_csv(datapath('equalriskasset.csv'),  index_col=['date'], parse_dates=['date'])
#     if not startdate:
#         startdate = df_nav.index.min().strftime("%Y-%m-%d")

#     timing_ids = ['49101', '49201', '49301', '49401']
#     df_timing = database.asset_tc_timing_signal_load(
#         timing_ids, begin_date=startdate, end_date=enddate)

#     tasks = {
#         'largecap'       : [49101, 11],
#         'smallcap'       : [49101, 12],
#         'rise'           : [49101, 13],
#         'oscillation'    : [49101, 14],
#         'decline'        : [49101, 15],
#         'growth'         : [49101, 16],
#         'value'          : [49101, 17],
#         'SP500.SPI'      : [49201, 41],
#         'GLNC'           : [49301, 42],
#         'HSCI.HI'        : [49401, 43],
#         'convertiblebond': None,
#         'creditbond'     : None,
#         'money'          : None,
#         'ratebond'       : None,
#     }

#     data = {}
#     risk_mgr = RiskManagement.RiskManagement()
#     for asset in df_nav.columns:
#         #
#         # 计算风控仓位
#         #
#         if tasks[asset] is None:
#             continue

#         timing_id, category = tasks[asset]

#         df = pd.DataFrame({
#             'nav': df_nav[asset],
#             'timing': df_timing[timing_id].reindex(df_nav.index, method='pad')
#         })
        
#         df_new = risk_mgr.perform(asset, df)
        
#         df_new['rm_risk_mgr_id'] = optinst
#         df_new['rm_category'] = category
#         df_new = df_new.reset_index().set_index(['rm_risk_mgr_id', 'rm_category', 'rm_date'])
#         if not df_new.empty:
#             df_new = df_new.applymap("{:.0f}".format)
        
#         #
#         # 保存风控仓位到数据库
#         #
#         db = database.connection('asset')
#         t2 = Table('rm_risk_mgr_pos', MetaData(bind=db), autoload=True)
#         columns2 = [
#             t2.c.rm_risk_mgr_id,
#             t2.c.rm_category,
#             t2.c.rm_date,
#             t2.c.rm_action,
#             t2.c.rm_pos,
#         ]
#         s = select(columns2, (t2.c.rm_risk_mgr_id == optinst) & (t2.c.rm_category == category))
#         df_old = pd.read_sql(s, db, index_col=['rm_risk_mgr_id', 'rm_category', 'rm_date'], parse_dates=['rm_date'])
#         if not df_old.empty:
#             df_old = df_old.applymap("{:.0f}".format)

#         # 更新数据库
#         # print df_new.head()
#         # print df_old.head()
#         database.batch(db, t2, df_new, df_old, timestamp=False)

#     #
#     # 合并 markowitz 仓位 与 风控 结果
#     #
#     df_pos_markowitz = pd.read_csv(datapath('portfolio_position.csv'), index_col=['date'], parse_dates=['date'])

#     df_pos_riskmgr = database.asset_rm_risk_mgr_pos_load(optinst)

#     for column in df_pos_riskmgr.columns:
#         category = DFUtil.categories_name(column)
#         # if column < 20:
#         #     rmc = 11
#         # else:
#         #     rmc = column
#         rmc = column
#         # print "use column %d for category %s" % (rmc, category)
                
#         if category in df_pos_markowitz:
#             df_pos_tmp = df_pos_riskmgr[rmc].reindex(df_pos_markowitz.index)
#             df_pos_markowitz[category] = df_pos_markowitz[category] * df_pos_tmp

#     df_result = df_pos_markowitz.reset_index().set_index(['risk', 'date'])

#     #
#     # 调整货币的比例, 总和达到1
#     #
#     df_result['money'] = (1 - (df_result.sum(axis=1) - df_result['money']))
    
#     df_result.to_csv(datapath('riskmgr_position.csv'))

# def make_rm_risk_mgr_if_not_exist(id_):
#     db = database.connection('asset')
#     t2 = Table('rm_risk_mgr', MetaData(bind=db), autoload=True)
#     columns2 = [
#         t2.c.globalid,
#         t2.c.rm_inst_id,
#     ]
#     s = select(columns2, (t2.c.globalid == id_))
#     df = pd.read_sql(s, db, index_col=['globalid'])
#     if not df.empty:
#         return True
#     #
#     # 导入数据
#     #
#     row = {
#         'globalid': id_, 'rm_inst_id': id_, 'created_at': func.now(), 'updated_at': func.now()
#     }
#     t2.insert(row).execute()

#     return True

import numpy as np
import matplotlib.pyplot as plt
