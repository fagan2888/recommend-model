#coding=utf8


import string
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import os
import time
import logging
import re
import Const
import DFUtil
import DBData
import util_numpy as npu
import Portfolio as PF
from TimingWavelet import TimingWt
import multiprocessing
from multiprocessing import Manager
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, asset_mz_markowitz, asset_mz_markowitz_alloc, asset_mz_markowitz_argv,  asset_mz_markowitz_asset, asset_mz_markowitz_criteria, asset_mz_markowitz_nav, asset_mz_markowitz_pos, asset_mz_markowitz_sharpe, asset_wt_filter_nav
from db import asset_ra_pool, asset_ra_pool_nav, asset_rs_reshape, asset_rs_reshape_nav, asset_rs_reshape_pos
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates, base_exchange_rate_index_nav, asset_ra_bl
from RiskParity import cal_weight
from util import xdict
from util.xdebug import dd
from asset import Asset, WaveletAsset
from allocate import Allocate
from trade_date import ATradeDate
from view import View
import Financial as fin
from ipdb import set_trace
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('seaborn')


import traceback, code


logger = logging.getLogger(__name__)



class AvgAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(AvgAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):
        ws = [1.0 / len(df_inc.columns)] * len(df_inc.columns)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


class KellyAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(KellyAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):
        ws = fin.kelly_alloc(df_inc, bound)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


class MvpAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(MvpAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):
        ws = fin.mvp_alloc(df_inc, bound)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


class MzAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(MzAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):
        risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


class RpAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(RpAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):
        # risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
        # ws = dict(zip(df_inc.columns.ravel(), ws))
        V = df_inc.cov().values*1e4
        lb = 0.03
        hb = (1-lb)/7
        risk_budget = [hb,hb,lb,hb,hb,hb,hb,hb]
        # weight = PF.riskparity(df_inc)
        weight = cal_weight(V, risk_budget)
        ws = dict(zip(df_inc.columns.ravel(), weight))
        return ws


class MzBlAllocate(Allocate):


    def __init__(self, globalid, assets, views, reindex, lookback, period = 1, bound = None):
        super(MzBlAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        self.__views = views

    def allocate_algo(self, day, df_inc, bound):
        P, eta, alpha = self.load_bl_view(day, list(self.assets.keys()))
        risk, returns, ws, sharpe = PF.markowitz_r_spe_bl(df_inc, P, eta, alpha, bound)
        ws = dict(zip(df_inc.columns.ravel(), ws))

        return ws


    def load_bl_view(self, day, asset_ids):

        confidences = []
        view = pd.Series(0, index = asset_ids)
        for asset_id in asset_ids:
            view.loc[asset_id] = self.__views[asset_id].view(day)
            confidences.append(self.__views[asset_id].confidence)

        eta = np.array(abs(view[view!=0]))
        P = np.diag(np.sign(view))
        P = np.array([i for i in P if i.sum()!=0])
        if eta.size == 0:
            P = None
            eta = np.array([])

        return P, eta, np.mean(confidences)


class MzBootAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzBootAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        if cpu_count is None:
            count = multiprocessing.cpu_count() / 2
            cpu_count = count if count > 0 else 1
            self.__cpu_count = cpu_count
        else:
            self.__cpu_count = cpu_count
        self.__bootstrap_count = bootstrap_count


    def allocate_algo(self, day, df_inc, bound):
        risk, returns, ws, sharpe = PF.markowitz_bootstrape(df_inc, bound, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws



class MzBootBlAllocate(MzBlAllocate):


    def __init__(self, globalid, assets, views, reindex, lookback, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzBootBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, period, bound)
        if cpu_count is None:
            count = multiprocessing.cpu_count() / 2
            cpu_count = count if count > 0 else 1
            self.__cpu_count = cpu_count
        else:
            self.__cpu_count = cpu_count
        self.__bootstrap_count = bootstrap_count


    def allocate_algo(self, day, df_inc, bound):
        P, eta, alpha = self.load_bl_view(day, list(self.assets.keys()))
        risk, returns, ws, sharpe = PF.markowitz_bootstrape_bl(df_inc, P, eta, alpha, bound, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


class MzMomAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(MzMomAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):
        # risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
        # ws = dict(zip(df_inc.columns.ravel(), ws))
        df_ret = df_inc.tail(1).sum()
        df_ret = df_ret.sort_values()
        df_ret.iloc[-5:] = 0.2
        df_ret.iloc[:-5] = 0.0

        ws = df_ret.to_dict()
        return ws


class MzMomAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(MzMomAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):
        # risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
        # ws = dict(zip(df_inc.columns.ravel(), ws))
        df_ret = df_inc.tail(1).sum()
        df_ret = df_ret.sort_values()
        df_ret.iloc[-5:] = 0.2
        df_ret.iloc[:-5] = 0.0

        ws = df_ret.to_dict()
        return ws


class MzMRAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(MzMRAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        self.amount = self.load_amount()
        self.volume = self.load_volume()


    def allocate_algo(self, day, df_inc, bound):
        # risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
        # ws = dict(zip(df_inc.columns.ravel(), ws))
        # df_amount = self.amount.loc[day]

        df_ret = df_inc.tail(1).sum()
        df_volume = self.volume.loc[day]

        df_ret_rank = pd.Series(data = rankdata(df_ret), index = df_ret.index)
        df_volume_rank = pd.Series(data = -rankdata(df_volume), index = df_volume.index)
        df_rank = df_ret_rank + df_volume_rank
        df_rank = df_rank.sort_values()
        alloc_num = 5

        df_rank.iloc[-alloc_num:] = 1.0 / alloc_num
        df_rank.iloc[:-alloc_num] = 0.0

        ws = df_rank.to_dict()
        return ws


    def allocate_algo_le(self, day, df_inc, bound, pre_ws):
        # risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
        # ws = dict(zip(df_inc.columns.ravel(), ws))
        # df_amount = self.amount.loc[day]

        alloc_num = 5
        df_ret = df_inc.tail(1).sum()
        df_volume = self.volume.loc[day]

        df_ret_rank = pd.Series(data = rankdata(df_ret), index = df_ret.index)
        df_volume_rank = pd.Series(data = -rankdata(df_volume), index = df_volume.index)
        df_rank = df_ret_rank + df_volume_rank
        df_rank = df_rank.sort_values()


        if pre_ws is None:
            df_ws = df_rank
            df_ws.iloc[-alloc_num:] = 1.0 / alloc_num
            df_ws.iloc[:-alloc_num] = 0.0
            ws = df_ws.to_dict()
        else:
            pre_factors = [k for (k,v) in pre_ws.iteritems() if v > 0]
            retain_factors = df_rank.loc[pre_factors].sort_values().tail(alloc_num-1).index.values
            for factor in df_rank.index[::-1]:
                if (factor not in retain_factors) and len(retain_factors) < alloc_num:
                    retain_factors = np.append(retain_factors, factor)
            df_ws = df_rank
            df_ws.loc[retain_factors] = 1.0 / alloc_num
            df_ws.loc[df_ws.index.difference(retain_factors)] = 0.0
            ws = df_ws.to_dict()
        return ws


    def load_amount(self):
        reindex = self.index
        data = {}
        for asset_id in self.assets.keys():
            data[asset_id] = base_ra_index_nav.load_ra_amount(asset_id, reindex = reindex)
        df_amount = pd.DataFrame(data).fillna(method='pad')

        return df_amount


    def load_volume(self):
        reindex = self.index
        data = {}
        for asset_id in self.assets.keys():
            data[asset_id] = base_ra_index_nav.load_ra_volume(asset_id, reindex = reindex)
        df_volume = pd.DataFrame(data).fillna(method='pad')

        return df_volume


class MzMRRevAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(MzMRRevAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        self.amount = self.load_amount()
        self.volume = self.load_volume()


    def allocate_algo(self, day, df_inc, bound):
        # risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
        # ws = dict(zip(df_inc.columns.ravel(), ws))
        # df_amount = self.amount.loc[day]

        df_ret = df_inc.tail(1).sum()
        df_volume = self.volume.loc[day]

        df_ret_rank = pd.Series(data = rankdata(df_ret), index = df_ret.index)
        df_volume_rank = pd.Series(data = -rankdata(df_volume), index = df_volume.index)
        df_rank = df_ret_rank + df_volume_rank
        df_rank = df_rank.sort_values()

        df_rank.iloc[5:] = 0.0
        df_rank.iloc[:5] = 0.2

        ws = df_rank.to_dict()
        return ws


    def load_amount(self):
        reindex = self.index
        data = {}
        for asset_id in self.assets.keys():
            data[asset_id] = base_ra_index_nav.load_ra_amount(asset_id, reindex = reindex)
        df_amount = pd.DataFrame(data).fillna(method='pad')

        return df_amount


    def load_volume(self):
        reindex = self.index
        data = {}
        for asset_id in self.assets.keys():
            data[asset_id] = base_ra_index_nav.load_ra_volume(asset_id, reindex = reindex)
        df_volume = pd.DataFrame(data).fillna(method='pad')

        return df_volume


class SptAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(SptAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):
        kde = KernelDensity(bandwidth = 0.0001)
        kde.fit(df_inc)
        roots = []
        for root in range(150):
            roots.append(kde.sample(60, random_state = root))

        w0 = [1.0 / len(df_inc.columns)] * len(df_inc.columns)
        cons = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': long_only}
        )
        # res = minimize(spt_objective, w0, args = [roots], method = 'SLSQP', constraints=cons, options={'disp':False, 'eps':0.01})
        res = minimize(spt_objective, w0, args = [roots], method = 'SLSQP', constraints=cons, options={'disp':False, 'eps':0.01})
        ws = res.x
        ws = dict(zip(df_inc.columns.ravel(), ws))
        print ' ', 1 - res.fun

        return ws


def spt_objective(x, pars):

    roots = pars[0]
    count = 0.0
    fail = 0.0
    for root in roots:
        count += 1
        ret = np.dot(root, x)
        nav = (1+ret).cumprod()
        # ret = root[:, 0]
        # nav = (1+ret).cumprod()
        # fig = plt.figure(figsize = (15,6))
        # ax = fig.add_subplot(111)
        # ax.plot(nav)
        # plt.savefig('/home/yaojiahui/Desktop/nav_%s.png'%datetime.now().microsecond)
        loss = min(nav) - 1
        tret = nav[-1] - 1
        # mdd = maxdd(nav)
        # print 'loss:', loss
        # print 'tret:',tret

        if loss < -0.01 or tret < 0.0:
        # if mdd < -0.01 or tret < 0.01:
            fail += 1
    # print x

    return fail/count


def sharpe_objective(x, pars):

    roots = pars[0]
    count = 0.0
    fail = 0.0
    for root in roots:
        count += 1
        ret = np.dot(root, x)
        # nav = (1+ret).cumprod()
        sharpe = ret.mean()/ret.std()

        if sharpe < 0.1:

        # if mdd < -0.01 or tret < 0.01:
            fail += 1
    # print x

    return fail/count

def long_only(x):
    return x

def maxdd(nav):
    return pd.Series(nav).rolling(len(nav), min_periods = 1).apply(lambda x: x[-1]/max(x) - 1).min()


if __name__ == '__main__':

    asset = Asset('120000001')
    print asset.nav().tail()

    asset = WaveletAsset('120000013', 2)
    print asset.nav('1900-01-01', datetime.now()).tail()

    trade_date = ATradeDate.week_trade_date(begin_date = '2012-01-01')
    print trade_date[-5:]

    asset_globalids = ['120000001', '120000002', '120000013', '120000014', '120000015']
    assets = {}
    for asset_id in asset_globalids:
        assets[asset_id] = Asset(asset_id)
        #assets[asset_id] = WaveletAsset(asset_id, 2)

    allocate = AvgAllocate('ALC.000001', assets, trade_date, 14)
    #print allocate.allocate().tail()

    allocate = MzAllocate('ALC.000002', assets, trade_date, 14)
    #print allocate.allocate().tail()

    allocate = MzBootAllocate('ALC.000002', assets, trade_date, 14)
    #print allocate.allocate().tail()


    view_df = View.load_view('BL.000001')

    asset_globalids = ['120000001', '120000002', 'ERI000001', '120000014', 'ERI000002']
    assets = {}
    for asset_id in asset_globalids:
        assets[asset_id] = Asset(asset_id)

    views = {}
    for asset_id in asset_globalids:
        views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = 0.5) if asset_id in view_df.columns else View(None, asset_id, confidence = 0.5)

    allocate = MzBootBlAllocate('ALC.000002', assets, views, trade_date, 26)
    print allocate.allocate().tail()
