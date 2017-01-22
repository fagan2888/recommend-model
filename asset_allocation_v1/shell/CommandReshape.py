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
import database
import DFUtil
import asset_mz_reshape
import util_numpy as npu

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from tabulate import tabulate


import traceback, code

logger = logging.getLogger(__name__)

@click.group()  
@click.pass_context
def reshape(ctx):
    '''reshape group
    '''
    pass



@reshape.command(name='import')
@click.option('--id', 'optid', type=int, help=u'specify reshape id')
@click.option('--name', 'optname', type=int, help=u'specify reshape name')
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
        df_existed = asset_rs_reshape.load([str(optid)])
        if not df_existed.empty:
            s = 'reshape instance [%d] existed' % optid
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

        max_id = asset_rs_reshape.max_id_between(between_min, between_max)
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
    rs_reshape = Table('rs_reshape', metadata, autoload=True)
    rs_reshape_pos = Table('rs_reshape_pos', metadata, autoload=True)
    rs_reshape_nav = Table('rs_reshape_nav', metadata, autoload=True)

    #
    # 处理替换
    #
    if optreplace:
        rs_reshape.delete(rs_reshape.c.globalid == optid).execute()
        rs_reshape_pos.delete(rs_reshape_pos.c.rs_reshape_id == optid).execute()
        rs_reshape_nav.delete(rs_reshape_nav.c.rs_reshape_id == optid).execute()

    now = datetime.now()
    #
    # 导入数据
    #
    row = {
        'globalid': optid, 'rs_type':opttype, 'rs_name': optname,
        'rs_pool': '', 'rs_reshape': '', 'created_at': func.now(), 'updated_at': func.now()
    }
    rs_reshape.insert(row).execute()

    df = pd.read_csv(csv, parse_dates=['date'])
    df['risk'] = (df['risk'] * 10).astype(int)
    renames = dict(
        {'date':'rs_date', 'risk':'rs_alloc_id'}.items() + DFUtil.categories_types(as_int=True).items()
    )
    df = df.rename(columns=renames)
    df['rs_reshape_id'] = optid

    df.set_index(['rs_reshape_id', 'rs_alloc_id', 'rs_date'], inplace=True)

    # 四舍五入到万分位
    df = df.round(4)
    # 过滤掉过小的份额
    df[df.abs() < 0.0009999] = 0
    # 补足缺失
    df = df.apply(npu.np_pad_to, raw=True, axis=1)
    # 过滤掉相同
    df = df.groupby(level=(0,1), group_keys=False).apply(DFUtil.filter_same_with_last)

    df.columns.name='rs_asset'
    df_tosave = df.stack().to_frame('rs_ratio')
    df_tosave = df_tosave.loc[df_tosave['rs_ratio'] > 0, ['rs_ratio']]
    if not df_tosave.empty:
        database.number_format(df_tosave, columns=['rs_ratio'], precision=4)
    
    df_tosave['updated_at'] = df_tosave['created_at'] = now

    df_tosave.to_sql(rs_reshape_pos.name, db, index=True, if_exists='append', chunksize=500)

    if len(df_tosave.index) > 1:
        logger.info("insert %s (%5d) : %s " % (rs_reshape_pos.name, len(df_tosave.index), df_tosave.index[0]))

    click.echo(click.style("import complement! instance id [%s]" % (optid), fg='green'))
    
    return 0

# @reshape.command()
# @click.option('--datadir', '-d', type=click.Path(exists=True), default='./tmp', help=u'dir used to store tmp data')
# @click.option('--id', 'optid', help=u'fund pool id to update')
# @click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
# @click.option('--online/--no-online', 'optonline', default=False, help=u'include online instance')
# @click.pass_context
# def signal(ctx, datadir, optid, optlist, optonline):
#     '''calc timing signal  for timing instance
#     '''
#     Const.datadir = datadir

#     if optid is not None:
#         timings = [s.strip() for s in optid.split(',')]
#     else:
#         timings = None

#     xtypes = None
#     if optonline == False:
#         xtypes = [1]

#     df_timing = database.asset_tc_timing_load(timings, xtypes)

#     if optlist:

#         df_timing['tc_name'] = df_timing['tc_name'].map(lambda e: e.decode('utf-8'))
#         print tabulate(df_timing, headers='keys', tablefmt='psql')
#         return 0
    
#     with click.progressbar(length=len(df_timing), label='update signal') as bar:
#         for _, timing in df_timing.iterrows():
#             bar.update(1)
#             signal_update(timing)

# def signal_update(timing):
#     '''calc timing signal for singe timing instance
#     '''
#     #
#     # 加载OHLC数据
#     #
#     timing_id = timing['globalid']
#     yesterday = (datetime.now() - timedelta(days=1)); 
#     enddate = yesterday.strftime("%Y-%m-%d")        
        
#     df_nav = database.base_ra_index_nav_load_ohlc(
#         timing['tc_index_id'], begin_date=timing['tc_begin_date'], end_date=enddate, mask=[0, 2])
        
#     # df_nav = pd.read_csv(datapath('000300_gftd_result.csv'),  index_col=['date'], parse_dates=['date'], usecols=['date', 'open', 'high', 'low', 'close'])
#     df_nav.rename(columns={'ra_open':'tc_open', 'ra_high':'tc_high', 'ra_low':'tc_low', 'ra_close':'tc_close'}, inplace=True)
#     df_nav.index.name='tc_date'
   
#     # risk_mgr = RiskManagement.RiskManagement()
#     df_new = TimingGFTD().timing(df_nav)
#     df_new['tc_timing_id'] = timing_id
#     df_new = df_new.reset_index().set_index(['tc_timing_id', 'tc_date'])

#     # print df_new[df_new['tc_stop'].isnull()].head()
#     num_signal = df_new['tc_signal'].rolling(2, 1).apply(lambda x: 1 if x[-1] != x[0] else 0).sum()
    
#     formaters = ['tc_close', 'tc_open', 'tc_high', 'tc_low', 'tc_recording_high', 'tc_recording_low', 'tc_stop_high', 'tc_stop_low']

#     if not df_new.empty:
#         df_new = database.number_format(df_new, columns=formaters, precision=4)

#     #
#     # 保存择时结果到数据库
#     #
#     db = database.connection('asset')
#     t2 = Table('tc_timing_scratch', MetaData(bind=db), autoload=True)
#     columns2 = [
#         t2.c.tc_timing_id,
#         t2.c.tc_date,
#         t2.c.tc_open,
#         t2.c.tc_high,
#         t2.c.tc_low,
#         t2.c.tc_close,
#         t2.c.tc_ud,
#         # t2.c.tc_ud_flip,
#         t2.c.tc_ud_acc,
#         t2.c.tc_buy_start,
#         # t2.c.tc_buy_kstick,
#         t2.c.tc_buy_count,
#         t2.c.tc_buy_signal,
#         t2.c.tc_sell_start,
#         # t2.c.tc_sell_kstick,
#         t2.c.tc_sell_count,
#         t2.c.tc_sell_signal,
#         t2.c.tc_action,
#         t2.c.tc_recording_high,
#         t2.c.tc_recording_low,
#         t2.c.tc_signal,
#         t2.c.tc_stop_high,
#         t2.c.tc_stop_low,
#     ]
#     s = select(columns2, (t2.c.tc_timing_id == timing_id))
#     df_old = pd.read_sql(s, db, index_col=['tc_timing_id', 'tc_date'], parse_dates=['tc_date'])
#     if not df_old.empty:
#         df_old = database.number_format(df_old, columns=formaters, precision=4)

#     # 更新数据库
#     database.batch(db, t2, df_new, df_old, timestamp=False)
#     print "total signal: %d, %.2f/year" % (num_signal, num_signal * 250/len(df_new))

#     # 更新tc_timing_signal
#     df_new = df_new[['tc_signal']]
#     t3 = Table('tc_timing_signal', MetaData(bind=db), autoload=True)
#     columns3 = [
#         t3.c.tc_timing_id,
#         t3.c.tc_date,
#         t3.c.tc_signal,
#     ]
#     s = select(columns3, (t3.c.tc_timing_id == timing_id))
#     df_old = pd.read_sql(s, db, index_col=['tc_timing_id', 'tc_date'], parse_dates=['tc_date'])

#     # 更新数据库
#     database.batch(db, t3, df_new, df_old, timestamp=False)

# @timing.command()
# @click.option('--id', 'optid', help=u'ids of fund pool to update')
# @click.option('--list/--no-list', 'optlist', default=False, help=u'list pool to update')
# @click.pass_context
# def nav(ctx, optid, optlist):
#     ''' calc pool nav and inc
#     '''
#     if optid is not None:
#         timings = [s.strip() for s in optid.split(',')]
#     else:
#         timings = None

#     df_timing = database.asset_tc_timing_load(timings)

#     if optlist:

#         df_timing['tc_name'] = df_timing['tc_name'].map(lambda e: e.decode('utf-8'))
#         print tabulate(df_timing, headers='keys', tablefmt='psql')
#         return 0
    
#     with click.progressbar(length=len(df_timing), label='update nav') as bar:
#         for _, timing in df_timing.iterrows():
#             bar.update(1)
#             nav_update(timing)

# def nav_update(timing):
#     timing_id = timing['globalid']
#     # 加载择时信号
#     df_position = database.asset_tc_timing_scratch_load_signal(timing_id)
#     # 构建仓位
#     df_position.loc[df_position[timing_id] < 1, timing_id] = 0
    
#     # 加载基金收益率
#     min_date = df_position.index.min()
#     #max_date = df_position.index.max()
#     max_date = (datetime.now() - timedelta(days=1)) # yesterday


#     df_nav = database.base_ra_index_nav_load_series(
#         timing['tc_index_id'], begin_date=min_date, end_date=max_date, mask=0)
#     df_inc = df_nav.pct_change().fillna(0.0).to_frame(timing_id)

#     # 计算复合资产净值
#     df_nav_portfolio = DFUtil.portfolio_nav(df_inc, df_position, result_col='portfolio')

#     df_result = df_nav_portfolio[['portfolio']].rename(columns={'portfolio':'tc_nav'}).copy()
#     df_result.index.name = 'tc_date'
#     df_result['tc_inc'] = df_result['tc_nav'].pct_change().fillna(0.0)
#     df_result['tc_timing_id'] = timing['globalid']
#     df_result = df_result.reset_index().set_index(['tc_timing_id', 'tc_date'])
    
#     df_new = database.number_format(df_result, columns=['tc_nav', 'tc_inc'], precision=6)

#     # 加载旧数据
#     db = database.connection('asset')
#     t2 = Table('tc_timing_nav', MetaData(bind=db), autoload=True)
#     columns2 = [
#         t2.c.tc_timing_id,
#         t2.c.tc_date,
#         t2.c.tc_nav,
#         t2.c.tc_inc,
#     ]
#     stmt_select = select(columns2, (t2.c.tc_timing_id == timing['globalid']))
#     df_old = pd.read_sql(stmt_select, db, index_col=['tc_timing_id', 'tc_date'], parse_dates=['tc_date'])
#     if not df_old.empty:
#         df_old = database.number_format(df_old, columns=['tc_nav', 'tc_inc'], precision=6)

#     # 更新数据库
#     database.batch(db, t2, df_new, df_old, timestamp=False)
    
#     #print df_result.head()

#     # df_result.to_csv(datapath('riskmgr_result.csv'))

