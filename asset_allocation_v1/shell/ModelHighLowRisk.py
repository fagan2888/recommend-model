#coding=utf8


import os
import string
import click
import logging
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import Portfolio as PF
import DBData
import DFUtil
import AllocationData
import random
import threading
import Queue
import thread
import multiprocessing


from Const import datapath

logger = logging.getLogger(__name__)

def get_columns(key, excluded):
    if excluded is None:
        excluded = []
        
    columns = {
        'low':[e for e in ['ratebond','creditbond'] if e not in excluded],
        'high':[e for e in ['largecap', 'smallcap', 'rise', 'decline', 'growth', 'value', 'SP500.SPI', 'GLNC', 'HSCI.HI'] if e not in excluded]
        #'high':[e for e in ['largecap', 'smallcap','SP500.SPI', 'GLNC', 'HSCI.HI'] if e not in excluded]
    }

    return columns.get(key)
# def get_columns(key, includeDate=False):
#     columns = {
#         'low':['ratebond','creditbond'],
#         'high':['largecap', 'smallcap', 'rise', 'decline', 'growth', 'value', 'SP500.SPI', 'GLNC', 'HSCI.HI']
#     }

#     return columns.get(key)


def m_markowitz(queue, random_index, df_inc, bound):
    for index in random_index:
        tmp_df_inc = df_inc.iloc[index]
        risk, returns, ws, sharpe = PF.markowitz_r_spe(tmp_df_inc, bound)
        queue.put(ws)


def asset_alloc_high_risk_per_day(day, lookback, df_inc=None, columns=None):
    '''perform asset allocation of high risk asset for single day
    '''
    if columns and 'date' not in columns:
        columns.insert(0, 'date')

    # 加载时间轴数据
    index = DBData.trade_date_lookback_index(end_date=day, lookback=lookback)

    # 加载数据
    if df_inc is None:
        df_nav  = pd.read_csv(datapath('equalriskasset.csv'), index_col='date', parse_dates=['date'], usecols=columns)
        df_inc  = df_nav.pct_change().fillna(0.0)

    #
    # 根据时间轴进行重采样
    #
    df_inc = df_inc.reindex(index, fill_value=0.0)
    
    #
    # 基于马克维茨进行资产配置
    #
    # uplimit   = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3]
    # downlimit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bound_set = {
        'largecap': {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'smallcap': {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'rise':     {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'decline':  {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'growth':   {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'value':    {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'SP500.SPI':{'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'GLNC':     {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
        'HSCI.HI':  {'downlimit': 0.0, 'uplimit': 1.0, 'sumlimit': False},
    }

    bound = []
    for asset in df_inc.columns:
        bound.append(bound_set[asset])

    risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
    #risk, returns, ws, sharpe = PF.markowitz_bootstrape(df_inc, bound)
    sr_result = pd.concat([
        pd.Series(ws, index=df_inc.columns),
        pd.Series((sharpe, risk, returns), index=['sharpe','risk', 'return'])
    ])

    return sr_result

def asset_alloc_low_risk_per_day(day, lookback, df_inc=None, columns=None):
    '''perform asset allocation of low risk asset for single day
    '''
    if columns and 'date' not in columns:
        columns.insert(0, 'date')

    # 加载时间轴数据
    index = DBData.trade_date_lookback_index(end_date=day, lookback=lookback)

    # 加载数据
    if df_inc is None:
        df_nav = pd.read_csv(datapath('labelasset.csv'), index_col='date', parse_dates=['date'], usecols=columns)
        df_inc  = df_nav.pct_change().fillna(0.0)
    #
    # 根据时间轴进行重采样
    #
    df_inc = df_inc.reindex(index, fill_value=0.0)
    
    #
    # 基于马克维茨进行资产配置
    #
    risk, returns, ws, sharpe = PF.markowitz_r(df_inc, None)

    sr_result = pd.concat([
        pd.Series(ws, index=df_inc.columns),
        pd.Series((sharpe, risk, returns), index=['sharpe','risk', 'return'])
    ])

    return sr_result


def asset_alloc_high_low(start_date, end_date=None, lookback=26, adjust_period=None, excluded=None):
    '''perform asset allocation with constant-risk + high_low model.
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
    # 计算每个调仓点的最新配置
    #
    df_high = pd.DataFrame(index=pd.Index([], name='date'), columns=get_columns('high', excluded))
    df_low =  pd.DataFrame(index=pd.Index([], name='date'), columns=get_columns('low', excluded))

    with click.progressbar(length=len(adjust_index), label='markowitz') as bar:
        for day in adjust_index:
            logger.debug("markowitz: %s", day.strftime("%Y-%m-%d"))
            # 高风险资产配置
            df_high.loc[day] = asset_alloc_high_risk_per_day(day, lookback, columns=get_columns('high', excluded))
            # 底风险资产配置
            df_low.loc[day] = asset_alloc_low_risk_per_day(day, lookback, columns=get_columns('low', excluded))
            bar.update(1)

    # df_high.index.name='date'
    # df_low.index.name='date'

    #
    # 保存高低风险配置结果
    #
    df_high.to_csv(datapath('high_position.csv'), index_label='date')
    df_low.to_csv(datapath('low_position.csv'), index_label='date')

    #
    # 计算高风险资产的资产净值
    #
    df_inc = DFUtil.load_inc_csv(datapath('equalriskasset.csv'), get_columns('high', excluded), index)
    #df_inc = DFUtil.load_inc_csv(datapath('labelasset.csv'), get_columns('high', excluded), index)
    df_nav_high = DFUtil.portfolio_nav(df_inc, df_high[get_columns('high', excluded)])
    df_nav_high.to_csv(datapath('high_nav.csv'), index_label='date')
    
    #
    # 计算低风险资产的资产净值
    #
    df_inc = DFUtil.load_inc_csv(datapath('labelasset.csv'), get_columns('low', excluded), index)
    df_nav_low = DFUtil.portfolio_nav(df_inc, df_low[get_columns('low', excluded)])
    df_nav_low.to_csv(datapath('low_nav.csv'), index_label='date')
        
    #
    # 混合后风险1-10的资产净值
    #
    # 高低风险净值增长率
    df_inc_high = df_nav_high.pct_change().fillna(0.0)
    df_inc_low = df_nav_low.pct_change().fillna(0.0)
    df_inc = pd.DataFrame({'high':df_inc_high['portfolio'], 'low':df_inc_low['portfolio']})
    # 高低风险配置比例
    dt = dict()
    for risk in range(1, 11):
        # 配置比例
        ratio_h  = (risk - 1) * 1.0 / 9
        ratio_l  = 1 - ratio_h
        # 按调仓日期,生成调仓矩阵
        data = [(ratio_h, ratio_l) for x in adjust_index]
        df_position = pd.DataFrame(data, index=adjust_index, columns=['high', 'low'])
        # 单个风险配置结果
        dt['%.1f' % (risk / 10.0)] = DFUtil.portfolio_nav(df_inc, df_position)

    #
    # 保存高低风险配置结果
    # 
    df_nav_result = pd.concat(dt, names=('risk', 'date'))
    df_nav_result.to_csv(datapath('portfolio_nav.csv'))
    #
    # 返回结果
    #
    return df_nav_result


if __name__ == '__main__':
    pass
