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



class RiskMgrLow(object):

    def __init__(self, empty=25, maxdd=-0.075, mindd=-0.05, period=252):
        self.maxdd = maxdd
        self.mindd = mindd
        self.empty = empty
        self.period = period

    def perform(self, asset, df_input):
        #
        df = pd.DataFrame({
            'nav' : df_input['nav'],
            'max': df_input['nav'].rolling(25).max(),
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
                nav = row['nav']
                _max = row['max']

                # print day, (_max - nav) / _max
                if (_max - nav) / _max >= 0.015:
                    status, empty_days, position, action = 1, 0, 0, 2

                #if row['inc2d'] < self.maxdd:
                #    status, empty_days, position, action = 1, 0, 0, 2
                #elif row['inc3d'] < self.maxdd:
                #    status, empty_days, position, action = 1, 0, 0, 3
                #elif row['cnfdn'] is not None and row['inc5d'] < row['cnfdn'] and row['inc5d'] < self.mindd:
                #    status, empty_days, position, action = 1, 0, 0, 5

                #
                # 根据当前的风控状态进行处理
                #
                if status == 0:
                    # 不在风控中
                    status, position, action = 0, 1, 0
                else:
                    # 风控中 (status == 1)
                    if empty_days >= self.empty:
                            status, position, action = 0, 1, 8 # 满仓
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
