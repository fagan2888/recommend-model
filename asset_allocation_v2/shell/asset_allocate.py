#coding=utf8


import string
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
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
from ipdb import set_trace

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, asset_mz_markowitz, asset_mz_markowitz_alloc, asset_mz_markowitz_argv,  asset_mz_markowitz_asset, asset_mz_markowitz_criteria, asset_mz_markowitz_nav, asset_mz_markowitz_pos, asset_mz_markowitz_sharpe, asset_wt_filter_nav
from db import asset_ra_pool, asset_ra_pool_nav, asset_rs_reshape, asset_rs_reshape_nav, asset_rs_reshape_pos,asset_rm_riskmgr_signal
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates, base_exchange_rate_index_nav, asset_ra_bl
from db.asset_stock_factor import *
from util import xdict
from util.xdebug import dd
from asset import Asset, WaveletAsset
from allocate import Allocate
from trade_date import ATradeDate
from view import View
from RiskParity import cal_weight
import util_optimize
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

import PureFactor
import traceback, code


logger = logging.getLogger(__name__)



class AvgAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(AvgAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):
        ws = [1.0 / len(df_inc.columns)] * len(df_inc.columns)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws



class MzAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(MzAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):
        risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


class MzRiskMgrAllocate(Allocate):


    def __init__(self, globalid, assets, assets_riskmgr, reindex, lookback, period = 1, bound = None):
        super(MzRiskMgrAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        self.assets_riskmgr = assets_riskmgr
        self.riskmgr_df = self.load_riskmgr(self.assets_riskmgr)
        riskmgr_df_diff = self.riskmgr_df.diff()
        self.riskmgr_df_diff = riskmgr_df_diff[riskmgr_df_diff < 0].dropna(how = 'all')
        self.riskmgr_days = self.riskmgr_df_diff.index[self.riskmgr_df_diff.index >= self.index[self.lookback - 1]]


    def allocate_algo(self, day, df_inc, bound):
        risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, bound)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws

    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        self.riskmgr_days.name = adjust_days.name
        adjust_days = adjust_days.append(self.riskmgr_days)
        adjust_days = adjust_days.drop_duplicates()
        adjust_days = adjust_days.sort_values()
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index = adjust_days, columns = asset_ids)

        s = 'perform %-12s' % self.__class__.__name__

        with click.progressbar(
                adjust_days, label=s.ljust(30),
                item_show_func=lambda x:  x.strftime("%Y-%m-%d") if x else None) as bar:

            for day in bar:

                logger.debug("%s : %s", s, day.strftime("%Y-%m-%d"))

                df_inc, bound = self.load_allocate_data(day, asset_ids)

                ws = self.allocate_algo(day, df_inc, bound)

                for asset_id in list(ws.keys()):
                    pos_df.loc[day, asset_id] = ws[asset_id]

        data = {}
        for asset_id in pos_df.columns:
            pos_ser = pos_df[asset_id]
            if asset_id in self.riskmgr_df.columns:
                riskmgr_diff_ser = self.riskmgr_df[asset_id].diff()
                riskmgr_diff_ser = riskmgr_diff_ser.reindex(adjust_days).fillna(0.0)
                pos = []
                dates = pos_ser.index
                for d in dates:
                    if riskmgr_diff_ser.loc[d] < 0:
                        pos_ser = pos_ser[pos_ser.index >= d]
                        pos.append(pos_ser.loc[d])
                    else:
                        pos_rolling_ser = pos_ser.rolling(4, min_periods = 1).mean()
                        pos.append(pos_rolling_ser.loc[d])
                data[asset_id] = pd.Series(pos, index = dates)
            else:
                data[asset_id] = pos_ser.rolling(4, min_periods = 1).mean()

        pos_df = pd.DataFrame(data).fillna(0.0)

        sr = 1.0 - pos_df.sum(axis=1)
        if (sr > 0.000099).any():
            pos_df['120000039'] = pos_df['120000039'] + sr

        return pos_df


    def load_riskmgr(self, assets_riskmgr):
        data = {}
        for asset_id in assets_riskmgr.keys():
            riskmgr_id = assets_riskmgr[asset_id]
            sr = asset_rm_riskmgr_signal.load_series(riskmgr_id)
            sr.index.name = 'mz_date'
            data[asset_id] = sr

        df = pd.DataFrame(data).fillna(method='pad')

        return df


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
            count = int(multiprocessing.cpu_count()) // 2
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
            count = int(multiprocessing.cpu_count()) // 2
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


class MzBootDownRiskAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzBootDownRiskAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self.__cpu_count = cpu_count
        else:
            self.__cpu_count = cpu_count
        self.__bootstrap_count = bootstrap_count


    def allocate_algo(self, day, df_inc, bound):
        df_inc[df_inc >= 0] = 0.0
        risk, returns, ws, sharpe = PF.markowitz_bootstrape(df_inc, bound, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        tdate = ATradeDate.trade_date()
        var = np.array([self.assets[code].origin_nav_sr.reindex(tdate).pct_change().loc[df_inc.index[-13]:df_inc.index[-1]].var() for code in df_inc.columns])
        ws = np.array(ws).ravel()
        ws = ws/var
        ws = ws/ws.sum()
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


class MzFixRiskBootAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, risk, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzFixRiskBootAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self.__cpu_count = cpu_count
        else:
            self.__cpu_count = cpu_count
        self.__bootstrap_count = bootstrap_count
        self.risk = risk


    def allocate_algo(self, day, df_inc, bound):
        #ts = [0.5 ** (t / len(df_inc)) for t in range(0, len(df_inc))]
        #ts.reverse()
        #df_inc = df_inc.mul(ts, axis = 0)
        risk, returns, ws, sharpe = PF.markowitz_bootstrape_fixrisk(df_inc, bound, self.risk, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        # risk, returns, ws, sharpe = PF.markowitz_fixrisk(df_inc, bound, self.risk)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws



class MzFixRiskBootWaveletAllocate(Allocate):

    def __init__(self, globalid, assets, wavelet_assets, reindex, lookback, risk, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzFixRiskBootWaveletAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self.__cpu_count = cpu_count
        else:
            self.__cpu_count = cpu_count
        self.__bootstrap_count = bootstrap_count
        self.risk = risk
        self.wavelet_assets = wavelet_assets


    def allocate_algo(self, day, df_inc, bound):
        wavelet_df_inc, wavelet_bound = self.load_wavelet_allocate_data(day, list(self.assets.keys()))
        df_inc = df_inc + wavelet_df_inc * 2
        risk, returns, ws, sharpe = PF.markowitz_bootstrape_fixrisk(df_inc, bound, self.risk, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


    def load_wavelet_allocate_data(self, day ,asset_ids):

        reindex = self.index[self.index <= day][-1 * self.lookback:]
        data = {}
        for asset_id in asset_ids:
            data[asset_id] = self.wavelet_assets[asset_id].nav(reindex = reindex)
        df_nav = pd.DataFrame(data).fillna(method='pad')
        df_inc  = df_nav.pct_change().dropna()
        bound = []
        for asset_id in df_inc.columns:
            bound.append(self.bound[asset_id].get_day_bound(day).to_dict())

        return df_inc, bound



class MzFixRiskBootWaveletBlAllocate(MzBlAllocate):

    def __init__(self, globalid, assets, wavelet_assets, views, reindex, lookback, risk, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzFixRiskBootWaveletBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self.__cpu_count = cpu_count
        else:
            self.__cpu_count = cpu_count
        self.__bootstrap_count = bootstrap_count
        self.risk = risk
        self.wavelet_assets = wavelet_assets


    def allocate_algo(self, day, df_inc, bound):
        wavelet_df_inc, wavelet_bound = self.load_wavelet_allocate_data(day, list(self.assets.keys()))
        df_inc = df_inc + wavelet_df_inc
        P, eta, alpha = self.load_bl_view(day, list(self.assets.keys()))
        risk, returns, ws, sharpe = PF.markowitz_bootstrape_bl_fixrisk(df_inc, P, eta, alpha ,bound, self.risk, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


    def load_wavelet_allocate_data(self, day ,asset_ids):

        reindex = self.index[self.index <= day][-1 * self.lookback:]
        data = {}
        for asset_id in asset_ids:
            data[asset_id] = self.wavelet_assets[asset_id].nav(reindex = reindex)
        df_nav = pd.DataFrame(data).fillna(method='pad')
        df_inc  = df_nav.pct_change().dropna()
        bound = []
        for asset_id in df_inc.columns:
            bound.append(self.bound[asset_id].get_day_bound(day).to_dict())

        return df_inc, bound



class MzFixRiskBootBlAllocate(MzBlAllocate):

    def __init__(self, globalid, assets, views, reindex, lookback, risk, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzFixRiskBootBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self.__cpu_count = cpu_count
        else:
            self.__cpu_count = cpu_count
        self.__bootstrap_count = bootstrap_count
        self.risk = risk


    def allocate_algo(self, day, df_inc, bound):
        P, eta, alpha = self.load_bl_view(day, list(self.assets.keys()))
        risk, returns, ws, sharpe = PF.markowitz_bootstrape_bl_fixrisk(df_inc, P, eta, alpha, bound, self.risk, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


class MzRiskMgrFixRiskBootAllocate(MzRiskMgrAllocate):

    def __init__(self, globalid, assets, assets_riskmgr, reindex, lookback, risk, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzRiskMgrFixRiskBootAllocate, self).__init__(globalid, assets,assets_riskmgr, reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self.__cpu_count = cpu_count
        else:
            self.__cpu_count = cpu_count
        self.__bootstrap_count = bootstrap_count
        self.risk = risk


    def allocate_algo(self, day, df_inc, bound):
        risk, returns, ws, sharpe = PF.markowitz_bootstrape_fixrisk(df_inc, bound, self.risk, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws



class MzRiskMgrFixRiskBootWaveletAllocate(MzRiskMgrAllocate):

    def __init__(self, globalid, assets, wavelet_assets, assets_riskmgr, reindex, lookback, risk, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzRiskMgrFixRiskBootWaveletAllocate, self).__init__(globalid, assets, assets_riskmgr ,reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self.__cpu_count = cpu_count
        else:
            self.__cpu_count = cpu_count
        self.__bootstrap_count = bootstrap_count
        self.risk = risk
        self.wavelet_assets = wavelet_assets


    def allocate_algo(self, day, df_inc, bound):
        wavelet_df_inc, wavelet_bound = self.load_wavelet_allocate_data(day, list(self.assets.keys()))
        df_inc = df_inc + wavelet_df_inc * 2
        #风控当天把收益率降低一倍，更多的降低仓位
        for asset_id in df_inc.columns:
            if asset_id in self.riskmgr_df.columns:
                if (day in self.riskmgr_df.index) and (self.riskmgr_df.loc[day, asset_id] < 1.0):
                    df_inc.loc[day, asset_id] = df_inc.loc[day, asset_id] * 2 if df_inc.loc[day, asset_id] < 0 else 0.0
        risk, returns, ws, sharpe = PF.markowitz_bootstrape_fixrisk(df_inc, bound, self.risk, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


    def load_wavelet_allocate_data(self, day ,asset_ids):

        reindex = self.index[self.index <= day][-1 * self.lookback:]
        if not (reindex[-1] == day):
            reindex = reindex.insert(len(reindex), day)[1:]
        data = {}
        for asset_id in asset_ids:
            data[asset_id] = self.wavelet_assets[asset_id].nav(reindex = reindex)
        df_nav = pd.DataFrame(data).fillna(method='pad')
        df_inc  = df_nav.pct_change().dropna()
        bound = []
        for asset_id in df_inc.columns:
            bound.append(self.bound[asset_id].get_day_bound(day).to_dict())

        return df_inc, bound



class MzSTPAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(MzSTPAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):

        kde = KernelDensity(bandwidth = 0.0001)
        kde.fit(df_inc)
        roots = []
        for root in range(1000):
            roots.append(kde.sample(250, random_state = root))

        w0 = [0 / len(df_inc.columns)] * len(df_inc.columns)
        w0[df_inc.columns.tolist().index('120000039')] = 1.0
        cons = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x : x}
        )

        res = minimize(self.spt_objective, w0, args = [roots], method = 'SLSQP', constraints=cons, options={'disp':False, 'eps':0.01})
        ws = res.x
        ws = dict(zip(df_inc.columns.ravel(), ws))
        print(ws)
        return ws


    def spt_objective(self, x, pars):

        roots = pars[0]
        count = 0.0
        fail = 0.0
        rets = []
        for root in roots:
            count += 1
            ret = np.dot(root, x)
            nav = (1+ret).cumprod()
            # ret = root[:, 0]
            # nav = (1+ret).cumprod()
            loss = min(nav) - 1
            tret = nav[-1] - 1
            rets.append(tret)
            mdd = self.maxdd(nav)
            # print 'loss:', loss
            # print 'tret:',tret

            if mdd < -0.15:
                fail += 1

        if fail / count > 0.01:
            return 1.0
        else:
            return -1.0 * np.mean(rets)


    def maxdd(self, nav):
        return pd.Series(nav).rolling(len(nav), min_periods = 1).apply(lambda x: x[-1]/max(x) - 1).min()


    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index = adjust_days, columns = asset_ids)

        s = 'perform %-12s' % self.__class__.__name__

        with click.progressbar(
                adjust_days, label=s.ljust(30),
                item_show_func=lambda x:  x.strftime("%Y-%m-%d") if x else None) as bar:

            for day in bar:

                logger.debug("%s : %s", s, day.strftime("%Y-%m-%d"))

                df_inc, bound = self.load_allocate_data(day, asset_ids)

                ws = self.allocate_algo(day, df_inc, bound)

                for asset_id in list(ws.keys()):
                    pos_df.loc[day, asset_id] = ws[asset_id]

        return pos_df


if __name__ == '__main__':

    asset = Asset('120000001')

    asset = WaveletAsset('120000013', 2)

    trade_date = ATradeDate.week_trade_date(begin_date = '2012-01-01')

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
