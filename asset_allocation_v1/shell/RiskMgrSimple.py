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
from ipdb import set_trace



def confidence(x, per):
    return stats.norm.ppf(per, x.mean(), x.std(ddof=1))


class RiskMgrSimple(object):

    def __init__(self, empty=5, maxdd=-0.075, mindd=-0.05, period=252):
        self.maxdd = maxdd
        self.mindd = mindd
        self.mindd2 = mindd
        self.empty = empty
        self.period = period
        self.ratio = 0.3

    def perform(self, asset, df_input):
        #
        # 计算回撤矩阵 和 0.97, 0.75置信区间
        #
        sr_inc = np.log(1+df_input['nav'].pct_change().fillna(0.0))*1000
        # sr_inc = df_input['nav'].pct_change().fillna(0.0)
        sr_inc2d = sr_inc.rolling(window=2).sum() # 2日收益率
        sr_inc3d = sr_inc.rolling(window=3).sum() # 3日收益率
        sr_inc5d = sr_inc.rolling(window=5).sum() # 5日收益率

        from arch import arch_model


        df = pd.DataFrame({
            'inc2d': sr_inc2d.fillna(0),
            'inc3d': sr_inc3d.fillna(0),
            'inc5d': sr_inc5d.fillna(0),
            'timing': df_input['timing'],
        })
        # set_trace()
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
        flag = 0

        result_status = {}
        result_pos = {} # 结果仓位
        result_act = {} # 结果动作
        with click.progressbar(length=len(df.index), label='riskmgr %-20s' % (asset)) as bar:
            for day, row in df.iterrows():
  
                # 是否启动风控

                #保证有足够多的参数用于拟合, 先跳过300个历史数据
                if sr_inc.loc[:day].size < 300:
                    pass
                else:
                    df_inc2d = sr_inc2d[day::-2][::-1].fillna(0)
                    df_inc3d = sr_inc3d[day::-3][::-1].fillna(0)
                    df_inc5d = sr_inc5d[day::-5][::-1].fillna(0)
                    model_2d = arch_model(df_inc2d, mean='ARX', p=1,o=0,q=1)
                    model_3d = arch_model(df_inc3d, mean='ARX', p=1,o=0,q=1)
                    model_5d = arch_model(df_inc5d, mean='ARX', p=1,o=0,q=1)
                    res_2d = model_2d.fit(update_freq=10, disp='off')
                    res_3d = model_3d.fit(update_freq=10, disp='off')
                    res_5d = model_5d.fit(update_freq=10, disp='off')
                    if flag != 1:
                        if row['inc2d'] < (res_2d.params['Const'] - 3 * res_2d.conditional_volatility[-1]):
                            status, empty_days, position, action = 1, 0, self.ratio, 2
                        elif row['inc3d'] < (res_3d.params['Const'] - 3 * res_3d.conditional_volatility[-1]):
                            status, empty_days, position, action = 1, 0, self.ratio, 3
                        
                    if row['inc5d'] < (res_5d.params['Const'] - 3 * res_5d.conditional_volatility[-1]):
                        status, empty_days, position, action = 2, 0, 0, 5
                        flag = 1

                #
                # 根据当前的风控状态进行处理
                #
                if status == 0:
                    # 不在风控中
                    status, position, action = 0, 1, 0
                else:
                    # 风控中 (status=1或2)
                    if empty_days >= self.empty:
                        #择时决定何时满仓
                        if row['timing'] == 1.0:
                            status, position, action = 0, 1, 8 # 择时满仓
                            if flag == 1:
                                flag = 0
                        else:
                            empty_days += 1
                            if flag == 1:
                                status, position, action = 2, 0, 7
                            else:
                                status, position, action = 1, self.ratio, 7 # 空仓等待择时加仓信号
                    else:
                        empty_days += 1
                        #if empty_days != 1:
                        #    status, position, action = 1, 0, 6 # 无条件空仓

                result_status[day] = status
                result_act[day] = action
                result_pos[day] = position
                #
                # 更新进度条
                #
                bar.update(1)
        
        df_result = pd.DataFrame({'rm_pos': result_pos, 'rm_action': result_act})
        df_result.index.name = 'rm_date'
        
        # Regular calc winrate and exception
        # df = pd.DataFrame({'rm_pos': result_pos, 'rm_action': result_act, 'rm_status':result_status})
        # # df.index.name = 'rm_date'
        # indexes = [i for i in range(len(df)) if df.iloc[i].rm_action==8]
        # count_exception = [i for i in indexes if (df.iloc[i:i+7].rm_pos.sum()<7)]
        # idx_start = [i+1 for i in range(len(df)-1) if df.iloc[i].rm_pos == 1 and df.iloc[i+1].rm_pos==0]
        # idx_end = [i+1 for i in range(len(df)-1) if df.iloc[i].rm_pos == 0 and df.iloc[i+1].rm_pos==1]
        # if idx_end[0]<idx_start[0]:
        #     idx_end = idx_end[1:]
        # if len(idx_end) < len(idx_start):
        #     idx_end.append(len(df)-1)
        # inc = np.log(1+sr_inc)
        # count = np.array([inc.loc[df.iloc[i:j].index].sum() for i,j in zip(idx_start, idx_end)])
        # print "Risk Ctrl Triggered: " + str(count.size)
        # print "Winrate: " + str(count[count<0].size * 1.0 / count.size)
        # print len(count_exception)
        # print df.index[count_exception]
        # set_trace()

        #Modified
        df = pd.DataFrame({'rm_pos': result_pos, 'rm_action': result_act, 'rm_status':result_status})
        df.index.name = 'rm_date'
        indexes = [i for i in range(len(df)) if df.iloc[i].rm_action==8]
        count_exception = [i for i in indexes if (df.iloc[i:i+7].rm_pos.sum()<7)]
        status2=calcstatus(2,df)
        status1=calcstatus(1,df)
        inc = np.log(1+(sr_inc/1000.0))
        count2 = np.array([inc.iloc[i].sum() for i in status2])
        count1 = np.array([inc.iloc[i].sum() for i in status1])

        print "Lv1 Risk Triggered: " + str(count1.size)
        print "Lv1 Risk Ctrl Win: " + str(count1[count1<0].size)
        print "Lv2 Risk Triggered: " + str(count2.size)
        print "Lv2 Risk Ctrl Win: " + str(count2[count2<0].size)
        print "Overall Win Rate: " + str((count2[count2<0].size + 0.5* count1[count1<0].size) / (count2.size + 0.5*count1.size))




        return df_result;


def calcstatus(status, df):
    i = 1
    result = []
    while i < len(df):
        if df.iloc[i-1].rm_status != status and df.iloc[i].rm_status == status:
            tmp = []
            while i < len(df) and df.iloc[i].rm_status == status:
                tmp.append(i)
                i+=1
            result.append(tmp)
        else:
            i+=1 
    return result