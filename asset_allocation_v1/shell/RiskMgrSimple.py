# -*- coding: utf-8 -*-
"""
Created at Nov 23, 2016
Author: shengyitao
Contact: shengyitao@licaimofang.com
Company: LCMF
"""


import pandas as pd
import datetime
import numpy as np
import utils
import os
import sys
import click
import DFUtil
from scipy import stats
import random



def confidence(x, per):
    return stats.norm.ppf(per, x.mean(), x.std(ddof=1))


class RiskMgrSimple(object):

    def __init__(self, empty=5, maxdd=-0.075, mindd=-0.05, period=252):
        self.maxdd = maxdd
        self.mindd = mindd
        self.empty = empty
        self.period = period

    def perform(self, asset, df_input):
        #
        # 计算回撤矩阵 和 0.97, 0.75置信区间
        #
        sr_inc = df_input['nav'].pct_change().fillna(0.0)
        sr_inc2d = sr_inc.rolling(window=2).sum() # 2日收益率
        sr_inc3d = sr_inc.rolling(window=3).sum() # 3日收益率
        sr_inc5d = sr_inc.rolling(window=5).sum() # 5日收益率
       
        sr_cnfdn = sr_inc5d.rolling(window=self.period).apply(lambda x: stats.norm.ppf(0.01, x.mean(), x.std(ddof=1)))
        #sr_cnfdn = sr_cnfdn.shift(1)

        df = pd.DataFrame({
            'inc2d': sr_inc2d.fillna(0),
            'inc3d': sr_inc3d.fillna(0),
            'inc5d': sr_inc5d.fillna(0),
            'cnfdn': sr_cnfdn,
            'timing': df_input['timing'],
        })

        #
        # status: 0:不在风控中; 1:风控中
        # action: 0:无风控; 2:2日收益率触发风控; 3:3日收益率触发风控; 5:5日收益率触发风控; 6:无条件空仓; 7:无条件空仓结束等择时加仓信号; 8:择时满仓
        #
        # 风控逻辑:
        #
        #    1. if (五日收益率下跌到99%置信区间外 且 < -0.03) 或者 (2日或3日收益率 < -7.5%) 则启动风控, 开始空仓.
        #
        #    2. 启动风控后无条件空仓5天
        #    3. 5天后, 择时若给出全仓信号则全仓, 否则继续空仓直到择时给出全仓信号.
        #
        status, empty_days, action = 0, 0, 0

        result_pos = {} # 结果仓位
        result_act = {} # 结果动作
        with click.progressbar(length=len(df.index), label='riskmgr %-20s' % (asset)) as bar:
            for day, row in df.iterrows():
                #
                # 是否启动风控
                #
                if row['inc2d'] < self.maxdd:
                    status, empty_days, position, action = 1, 0, 0, 2
                elif row['inc3d'] < self.maxdd:
                    status, empty_days, position, action = 1, 0, 0, 3
                elif row['cnfdn'] is not None and row['inc5d'] < row['cnfdn']: #and row['inc5d'] < self.mindd:
                    status, empty_days, position, action = 1, 0, 0, 5

                print row
                #
                # 根据当前的风控状态进行处理
                #
                if status == 0:
                    # 不在风控中
                    status, position, action = 0, 1, 0
                else:
                    # 风控中 (status == 1)
                    if empty_days >= self.empty:
                        if row['timing'] == 1.0:
                            empty_days = 0
                            status, position, action = 0, 1, 8 # 择时满仓
                        else:
                            empty_days += 1
                            status, position, action = 1, 0, 7 # 空仓等待择时加仓信号
                    else:
                        empty_days += 1
                        #if empty_days != 1:
                        #    status, position, action = 1, 0, 6 # 无条件空仓


                result_act[day] = action
                result_pos[day] = position
                #
                # 更新进度条
                #
                bar.update(1)
        
        df_result = pd.DataFrame({'rm_pos': result_pos, 'rm_action': result_act})
        df_result.index.name = 'rm_date'

        return df_result;
