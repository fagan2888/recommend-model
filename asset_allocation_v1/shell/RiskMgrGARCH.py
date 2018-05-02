# coding=utf-8

import pandas as pd
import datetime
import numpy as np
import utils
import os
import sys
# sys.path.append(shell)
import click
import DFUtil
from scipy import stats
import random
from ipdb import set_trace
from pathos.multiprocessing import ProcessingPool as Pool
import RiskMgrGARCHHelper
import time
import pickle


class RiskMgrGARCHPrototype(object):
    def __init__(self, codes, target, tdates, df_nav, timing, vars_):
        self.empty = 5
        self.codes = codes
        self.target = target
        self.tdates = tdates
        self.df_nav = df_nav
        self.timings = timing
        self.vars = vars_
        #Cache for the status if the joint distribution got brokedown
        self.joints = {}

    def generate_df_for_garch(self, target):
        timing = self.timings[target].reindex(self.tdates[target])
        nav = self.df_nav.loc[:, target].reindex(self.tdates[target])
        inc = np.log(1+nav.dropna().pct_change().fillna(0))*100
        inc2d = inc.rolling(2).sum().fillna(0)
        inc3d = inc.rolling(3).sum().fillna(0)
        inc5d = inc.rolling(5).sum().fillna(0)
        return pd.DataFrame({'inc2d': inc2d,
                             'inc3d': inc3d,
                             'inc5d': inc5d,
                             'timing': timing})


class RiskMgrGARCH(RiskMgrGARCHPrototype):
    def __init__(self, codes, target, tdates, df_nav, timing, vars_):
        super(RiskMgrGARCH, self).__init__(codes, target, tdates, df_nav, timing, vars_)
        # self.ratio = 0

    def perform(self, target=None, disp=True):
        if target is None:
            target = self.target
        df = self.generate_df_for_garch(target)
        df_vars = self.vars[target]
        status, empty_days, action = 0, 0, 0
        result_status = {}
        result_pos = {} #结果仓位
        result_act = {} #结果动作
        for day, row in df.iterrows():
            if not (day in df_vars.index):
                pass
            else:
                # if status != 2:
                if row['inc2d'] < df_vars.loc[day]['var_2d']:
                    status, empty_days, position, action = 2, 0, 0, 2
                elif row['inc3d'] < df_vars.loc[day]['var_3d']:
                    status, empty_days, position, action = 2, 0, 0, 3
                elif row['inc5d'] < df_vars.loc[day]['var_5d']:
                    status, empty_days, position, action = 2, 0, 0, 5

            if status == 0:
                #不在风控中
                status, position, action = 0, 1, 0
            else:
                #风控中
                if empty_days >= self.empty:
                    #择时满仓
                    if row['timing'] == 1.0:
                        status, position, action = 0, 1, 8
                    else:
                        #空仓等待择时
                        empty_days += 1
                        if status == 2:
                            status, position, action = 2, 0, 7
                        # else:
                        #     status, position, action = 1, self.ratio, 7
                else:
                    empty_days += 1
            result_status[day] = status
            result_act[day] = action
            result_pos[day] = position

        df_result = pd.DataFrame({'rm_pos': result_pos, 'rm_action': result_act, 'rm_status': result_status})
        df_result.index.name = 'rm_date'
        if disp:
            self.calc_winrate(df_result, target)
        return df_result

    def calc_winrate(self, df_result, target):
        nav = self.df_nav[target].reindex(self.tdates[target])
        inc = np.log(1+nav.pct_change()).fillna(0)
        # status_half = calcstatus(1, df_result)
        status_full = calcstatus(2, df_result)
        count = lambda x: np.array([inc.iloc[i].sum() for i in x])
        # count_half = count(status_half)
        count_full = count(status_full)
        # print "Half triggered: %d" %  count_half.size
        # print "Half winned: %d" % count_half[count_half<0].size
        print "Full triggered: %d" % count_full.size
        print "Full winned: %d" % count_full[count_full<0].size



class RiskMgrMGARCH(RiskMgrGARCHPrototype):
    def __init__(self, codes, target, tdates, df_nav, timing, vars_, joints):
        super(RiskMgrMGARCH, self).__init__(codes, target, tdates, df_nav, timing, vars_)
        # self.ratio = 0
        self.joint_ratio = 0.5
        self.garch = RiskMgrGARCH(codes, target, tdates, df_nav, timing, vars_)
        self.joints = joints

    # def calc_joint(self, target=None):
    #     if target == self.target:
    #         return None
    #     if target is None:
    #         return pd.DataFrame({k: self.calc_joint(k) for k in self.codes if k != self.target}).fillna(False)
    #     if not (target in self.joints):
    #         print "Start multivariable GARCH fitting for %s" % target
    #         s_time = time.time()
    #         self.joints[target] = RiskMgrGARCHHelper.perform_joint(self.inc5d(target))
    #         print "Complete multivariable GARCH fitting! Elapsed time: %s" % time.strftime("%M:%S", time.gmtime(time.time()-s_time))
    #     return self.joints[target]

    def perform(self):
        # Calculate where the joint distribution get exceeded
        # df_joint = self.calc_joint()
        df_joint = self.joints
        others = df_joint.columns
        #取出其他资产仅用GARCH进行风控的结果
                            #按照联合分布的结果的日期reindex
                            #将缺失的交易日用前一天的风控结果填充
                            #将剩余的NaN以0 (False)填充
        df_result_garch = pd.DataFrame({k:self.garch.perform(k, disp=False)['rm_status'] for k in others}) \
                            .reindex(df_joint.index) \
                            .fillna(method='pad')    \
                            .fillna(0)
        #df_joint的row为日期, col为对应于目标资产的其他资产, entries为某日的目标资产与col对应资产的联合分布是否被击穿
        #例: 目标资产(self.target) => 61110403 (沪深300), col为['61110501'(中证500), '61110601'(标普500), '61110701'(恒生)]
        #则 df_joint.loc[day, '61110501']对应于指定日期(day)中, 沪深300与中证500构成的联合分布是否被击穿
        #df_result_garch的row, col同理, entries为某日的某资产仅被GARCH模型风控的状态
        #例: df_result_garch[day, '61110501']为指定日期(day)中, 中证500在GARCH模型下的风控状态(0 => 无风控, 1=> 风控一半(弃用), 2=>空仓)

        #对每个资产, 检查是否联合分布被击穿, 且其处于风控状态, 并最后求或(any(axis=1))
        if_joint_controlled = pd.DataFrame({k: (df_joint[k]!=0) & (df_result_garch[k]!=0) for k in others}).any(1)

        # Start the RiskMgr DF for the target asset
        df = self.generate_df_for_garch(self.target)
        df['joint_status'] = if_joint_controlled
        df_vars = self.vars[self.target]
        status, empty_days, action = 0, 0, 0
        flag = 0
        result_status = {}
        result_pos = {} # 结果仓位
        result_act = {} # 结果动作
        for day, row in df.iterrows():
            if not (day in df_vars.index):
                pass
            else:
                # if status < 3:
                if row['inc2d'] < df_vars.loc[day]['var_2d']:
                    status, empty_days, position, action = 2, 0, 0, 2
                elif row['inc3d'] < df_vars.loc[day]['var_3d']:
                    status, empty_days, position, action = 2, 0, 0, 3
                elif row['inc5d'] < df_vars.loc[day]['var_5d']:
                    status, empty_days, position, action = 2, 0, 0, 5

                if row['joint_status']:
                    if status == 0:
                        status, empty_days, position, action = 1, 0, self.joint_ratio, 4

                #联合分布是否被击穿有时可能没有数据, 判断这一天是否有对应数据
                # if day in df_joint.index:
                #     #判断当前天是否有被击穿
                #     if df_joint.loc[day].any():
                #         #取出所有联合分布被击穿的资产
                #         for idx in df_joint.loc[day][df_joint.loc[day]].keys():
                #             #若当前状态不在风控中
                #             if status == 0:
                #                 #若对应资产存在当前天被GARCH风控的结果(由于交易日的缘故, 可能不存在)
                #                 if day in df_result_garch[idx].index:
                #                     if df_result_garch[idx].loc[day]['rm_status'] != 0:
                #                         status, empty_days, position, action = 1, 0, self.joint_ratio, 4
                            # if status == 2:
                            #     if df_result_garch[idx].loc[day]['rm_status'] != 0:
                            #         status, empty_days, position, action = 4, 0, 0, 6
            if status == 0:
                #不在风控中
                status, position, action = 0, 1, 0
            else:
                #风控中
                if empty_days >= self.empty:
                    #择时满仓
                    if row['timing'] == 1.0:
                        status, position, action = 0, 1, 8
                    else:
                        #空仓等待择时
                        empty_days += 1
                        if status == 1:
                            status, position, action = 1, self.joint_ratio, 7
                        elif status == 2:
                            status, position, action = 2, 0, 7
                else:
                    empty_days += 1
            result_status[day] = status
            result_act[day] = action
            result_pos[day] = position

        df_result = pd.DataFrame({'rm_pos': result_pos, 'rm_action': result_act, 'rm_status': result_status})
        df_result.index.name = 'rm_date'
        self.calc_winrate_joint(df_result)
        return df_result

    def calc_winrate_joint(self, df_result):
        nav = self.df_nav[self.target].reindex(self.tdates[self.target])
        inc = np.log(1+nav.pct_change()).fillna(0)
        status_full = calcstatus(2, df_result)
        status_joint = calcstatus(1, df_result)
        count = lambda x: np.array([inc.iloc[i].sum() for i in x])
        count_full = count(status_full)
        count_joint = count(status_joint)
        print "Full triggered: %d" % count_full.size
        print "Full winned: %d" % count_full[count_full<0].size
        print "Joint half triggered: %d" % count_joint.size
        print "Joint half winned: %d" % count_joint[count_joint<0].size



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
