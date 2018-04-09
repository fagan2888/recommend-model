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
from RiskMgrMGARCH import RiskMgrMGARCH
import time

from db import *

mgarch = RiskMgrMGARCH()

# codes = {'sh300':{
#             'code': '120000001',
#             'timing': '21110100'},
#          'zz500':{
#              'code':'120000002',
#              'timing': '21110200'},
#          'sp500':{
#              'code':'120000013',
#              'timing':'21120200'},
#          'au9999':{
#              'code':'120000014',
#              'timing':'21400100'},
#          'hsi':{
#              'code':'120000015',
#              'timing':'21120500'}
#          }


class RPyGARCH(object):
    def __init__(self, codes, target, tdates, df_nav, timing):
        s_time = time.time()
        self.empty = 5
        self.ratio = 0.5
        self.joint = 0.3
        self.codes = codes
        self.target = target
        #Cache for VaRs calculated by R
        self.vars = {}
        #Cache for the status if the joint distribution got brokedown
        self.joints = {}
        self.tdates = tdates
        self.df_nav = df_nav
        self.timings = timing

    def timing(self, target=None):
        if target is None:
            target = self.target
        return self.timings[target].reindex(self.tdates[target])
    
    def inc(self, target=None, raw = False):
        if target is None or target == self.target:
            raw = True
            target = self.target
        if raw == True:
            nav = self.df_nav.loc[:, target]
        else:
            nav = self.df_nav.loc[:, [self.target, target]]
        return np.log(1+nav.dropna().pct_change()).fillna(0)*100

    def inc2d(self, target=None, raw=False):
        if target is None:
            target = self.target
        return self.inc(target, raw).rolling(2).sum().fillna(0)
    
    def inc3d(self, target=None, raw=False):
        if target is None:
            target = self.target
        return self.inc(target, raw).rolling(3).sum().fillna(0)

    def inc5d(self, target=None, raw=False):
        if target is None:
            target = self.target
        return self.inc(target, raw).rolling(5).sum().fillna(0)

    def generate_df_for_garch(self, target):
        return pd.DataFrame({'inc2d': self.inc2d(target, raw=True), 
                             'inc3d': self.inc3d(target, raw=True), 
                             'inc5d': self.inc5d(target, raw=True),
                             'timing': self.timing(target)})
    
    def calc_joint(self, target=None):
        if target == self.target:
            return None
        if target is None:
            return pd.DataFrame({k: self.calc_joint(k) for k in codes if k != self.target}).fillna(False)
        if not (target in self.joints):
            print "Start multivariable GARCH fitting for %s" % target
            s_time = time.time()
            self.joints[target] = mgarch.perform_joint(self.inc5d(target))
            print "Complete multivariable GARCH fitting! Elapsed time: %s" % time.strftime("%M:%S", time.gmtime(time.time()-s_time))
        return self.joints[target]

    def calc_garch(self, target, disp=True):
        print "Start Risk Ctrl for %s" % target
        df = self.generate_df_for_garch(target)
        s_time = time.time()
        #Check if the vars for target is in the cache
        if target in self.vars:
            df_vars = self.vars[target]
        else:
            df_vars = mgarch.perform_single(df.drop(columns=['timing']))
            self.vars[target] = df_vars
        print "Complete VaR calculation for %s! Elapsed time: %s" % (target, time.strftime("%M:%S", time.gmtime(time.time()-s_time)))
        status, empty_days, action = 0, 0, 0
        flag = 0
        result_status = {}
        result_pos = {} #结果仓位
        result_act = {} #结果动作
        for day, row in df.iterrows():
            if not (day in df_vars.index):
                pass
            else:
                if status != 2:
                    if row['inc2d'] < df_vars.loc[day]['VaR2d']:
                        status, empty_days, position, action = 1, 0, self.ratio, 2
                    elif row['inc3d'] < df_vars.loc[day]['VaR3d']:
                        status, empty_days, position, action = 1, 0, self.ratio, 3
                if row['inc5d'] < df_vars.loc[day]['VaR5d']:
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
                        else:
                            status, position, action = 1, self.ratio, 7
                else:
                    empty_days += 1
            result_status[day] = status
            result_act[day] = action
            result_pos[day] = position
        
        df_result = pd.DataFrame({'rm_pos': result_pos, 'rm_action': result_act, 'rm_status': result_status})
        df_result.index.name = 'rm_date'
        print "Accomplished Risk Ctrl for %s" % target
        if disp:
            self.calc_winrate(df_result)
        return df_result
     
    def calc_mult_garch(self):
        # Calculate where the joint distribution get exceeded
        df_joint = self.calc_joint()
        df_result_garch = {k:self.calc_garch(k, disp=False) for k in self.codes if k != self.target}
        # Start the RiskMgr DF for the target asset
        print "Start joint fitting"
        df = self.generate_df_for_garch(self.target)
        s_time = time.time()
        if self.target in self.vars:
            df_vars = self.vars[self.target]
        else:
            df_vars = mgarch.perform_single(df.drop(columns=['timing']))
            self.vars[self.target] = df_vars
        print "Complete VaR calculation for %s! Elapsed time: %s" % (self.target, time.strftime("%M:%S", time.gmtime(time.time()-s_time)))
        status, empty_days, action = 0, 0, 0
        flag = 0
        result_status = {}
        result_pos = {} # 结果仓位
        result_act = {} # 结果动作
        for day, row in df.iterrows():
            if not (day in df_vars.index):
                pass
            else:
                if status < 3:
                    if row['inc2d'] < df_vars.loc[day]['VaR2d']:
                        status, empty_days, position, action = 2, 0, self.ratio, 2
                    elif row['inc3d'] < df_vars.loc[day]['VaR3d']:
                        status, empty_days, position, action = 2, 0, self.ratio, 3
                
                if row['inc5d'] < df_vars.loc[day]['VaR5d']:
                    status, empty_days, position, action = 3, 0, 0, 5
                
                if day in df_joint.index:
                    if df_joint.loc[day].any():
                        for idx in df_joint.loc[day][df_joint.loc[day]].keys():
                            if status == 0:
                                if df_result_garch[idx].loc[day]['rm_status'] == 2:
                                    status, empty_days, position, action = 1, 0, self.joint, 4
                            if status == 2:
                                if df_result_garch[idx].loc[day]['rm_status'] == 2:
                                    status, empty_days, position, action = 4, 0, 0, 6
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
                            status, position, action = 1, self.joint, 7
                        elif status == 2:
                            status, position, action = 2, self.ratio, 7
                        elif status == 3:
                            status, position, action = 3, 0, 7
                        elif status == 4:
                            status, position, action = 4, 0, 7
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
        inc = np.log(1+self.df_nav[self.target].pct_change())
        status_half = calcstatus(2, df_result)
        status_full = calcstatus(3, df_result)
        status_joint_half = calcstatus(1, df_result)
        status_joint_full = calcstatus(4, df_result)
        count = lambda x: np.array([inc.iloc[i].sum() for i in x])
        count_half = count(status_half)
        count_full = count(status_full)
        count_joint_half = count(status_joint_half)
        count_joint_full = count(status_joint_full)
        print "Half triggered: %d" %  count_half.size
        print "Half winned: %d" % count_half[count_half<0].size
        print "Full triggered: %d" % count_full.size
        print "Full winned: %d" % count_full[count_full<0].size
        print "Joint half triggered: %d" % count_joint_half.size
        print "Joint half winned: %d" % count_joint_half[count_joint_half<0].size
        print "Joint full triggered: %d" % count_joint_full.size
        print "Joint full winned: %d" % count_joint_full[count_joint_full<0].size
    
    def calc_winrate(self, df_result):
        inc = np.log(1+self.df_nav[self.target].pct_change())
        status_half = calcstatus(1, df_result)
        status_full = calcstatus(2, df_result)
        count = lambda x: np.array([inc.iloc[i].sum() for i in x])
        count_half = count(status_half)
        count_full = count(status_full)
        print "Half triggered: %d" %  count_half.size
        print "Half winned: %d" % count_half[count_half<0].size
        print "Full triggered: %d" % count_full.size
        print "Full winned: %d" % count_full[count_full<0].size



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