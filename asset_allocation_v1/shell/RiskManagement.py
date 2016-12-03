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
import click
import DFUtil
from scipy import stats



class RiskManagement(object):

    def __init__(self):
        # 卖出置信区间
        self.threshhold_sell = 0.97
        # 买入置信区间
        self.threshhold_buy = 0.75

    def perform(self, df_nav, df_pos, df_timing):
        """Perform risk management base on df_nav and df_pos
        
        Keyword arguments:
        df_nav -- dataframe, each column is nav of single asset, indexed by dates
        df_pos -- dataframe, each column is position of single asset, indexed by date
        df_timing -- dataframe, each column is signal of timing, index by date

        Return:
        dataframe, echo column is the new position of the asset after risk management
        """        

        #
        # 计算回撤矩阵 和 0.97, 0.75置信区间
        #
        drawdown = []
        confidence97 = []
        confidence75 = []
        with click.progressbar(length=5, label='calc drawdown') as bar:
            for i in xrange(0, 5):
                bar.update(1)
                df_tmp = DFUtil.nav_drawdown_window(df_nav[i::5], 52)
                df_c97 = df_tmp.rolling(window=52, min_periods=52).apply(
                    lambda x: min(stats.norm.interval(0.97, x.mean(), x.std())))
                df_c97 = df_tmp.rolling(window=52, min_periods=52).apply(f97)
                df_c97 = df_c97.shift(1)
                df_c75 = df_tmp.rolling(window=52, min_periods=52).apply(
                    lambda x: min(stats.norm.interval(0.75, x.mean(), x.std())))
                df_c75 = df_c75.shift(1)

                drawdown.append(df_tmp)
                confidence97.append(df_c97)
                confidence75.append(df_c75)

        df_drawdown = pd.concat(drawdown).sort_index()
        df_confidence97 = pd.concat(confidence97).sort_index()
        df_confidence75 = pd.concat(confidence75).sort_index()

        result = {}
        for asset in df_nav.columns:
            start = df_pos.index.min()
            index = df_nav.truncate(before=start).index
            df = pd.DataFrame({
                'pos' :      df_pos[asset].reindex(index, method='pad'),
                'drawdown' : df_drawdown.loc[start:, asset],
                'c97' :      df_confidence97.loc[start:, asset],
                'c75' :      df_confidence75.loc[start:, asset],
                'timing':    df_timing[asset].reindex(index, method='pad'),
            })

            with click.progressbar(length=len(df.index), label='riskmgr %s' % (asset)) as bar:
                (pos, action) = self.control(df, bar)
                result[(asset, 'pos')] = pos
                result[(asset, 'action')] = action

        df_result = pd.DataFrame(result)
        df_result.index.name = 'date'
        df_result.columns.names = ['asset', 'p&a']
        print df_result.head()

        return df_result

    def control(self, df, bar=None):
        '''Perform core risk management actually

        Keyword arguments:
        df -- dataframe, columns=['pos', 'drawdown', 'c97', 'c75'] index='date'

        Return:
        dict, key=date, value=new_postion
        '''
        print "\n"
        print df.loc['2014-03']
        result_pos = {}
        result_action = {}
        status, empty_days, timing_status = (0, 0, 0)
        
        for day, row in df.iterrows():
            if bar is not None:
                bar.update(1)
                
            # 初始化变量
            pos, drawdown, c97, c75, timing = (
                row['pos'], row['drawdown'], row['c97'], row['c75'], row['timing'])
            #
            # 检测风控状态是否需要改变
            #
            if status == 0:
                if drawdown < c97:
                    # 不在风控中, 当前回撤在0.97置信区间外, 则启动风控
                    (status, timing_status, empty_days) = (1, 0, 0)
            else:
                if drawdown >= c75 and empty_days >= 5:
                    # 风控中, 如果当前回撤在0.75置信区间内, 则无条件结束风控
                    (status, timing_status, empty_days) = (0, 0, 0)
            #
            # 根据风控状态调整仓位
            #
            if status == 0:
                #
                # 不在风控周期内, 则持有原仓位
                #
                cur, action = (pos, 0)
            else:
                #
                # 风控中
                #
                if empty_days < 5:
                    # choice 1: 无条件空仓5天
                    cur, action = (0.0, 1)
                    empty_days += 1
                else:
                    #
                    # choice 2: 择时加仓和减仓
                    #
                    # 当前择时结果为买入 and 择时状态为空, 设置择时状态为买入
                    #
                    if timing == 1 and timing_status == 0:
                        timing_status = 1

                    if timing_status == 0:
                        # 尚未检测到择时信号, 则空仓
                        cur, action = (0.0, 2)
                    else:
                        # 当前回撤落在 0.97置信区间内:
                        if drawdown >= c97:
                            cur, action = (pos, 3)
                        elif drawdown <= c97 * 1.25:
                            cur, action = (0.0, 4)
                        else:
                            if last == 0:
                                cur, action = (0.0, 5)
                            else:
                                cur, action = (pos, 5)
                    
            # 记录当日风控结果, 并调整last
            result_pos[day] = last = cur
            result_action[day] = action

        return (result_pos, result_action)
        
        
