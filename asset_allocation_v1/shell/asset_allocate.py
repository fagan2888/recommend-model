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
from util import xdict
from util.xdebug import dd
from asset import Asset, WaveletAsset
from allocate import Allocate
from trade_date import ATradeDate
from view import View


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

    @property
    def cpu_count(self):
        return self.__cpu_count

    @property
    def bootstrap_count(self):
        return self.__bootstrap_count

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


class MzVolAdjAllocate(MzBootAllocate):

    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzVolAdjAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound, cpu_count, bootstrap_count)

    def allocate_algo(self, day, df_inc, bound):
        from ipdb import set_trace
        df_inc[df_inc>0] = 0
        tdate = ATradeDate.trade_date()
        #  var_res = {}
        #  for i in range(1, len(df_inc.index)):
            #  day = df_inc.index[i]
            #  tmp_var_res = {}
            #  for code in df_inc.columns:
                #  var = (self.assets[code].nav(reindex=tdate).pct_change()).loc[df_inc.index[i-1]:df_inc.index[i]].var()
                #  tmp_var_res[code] = var
            #  var_res[day]=tmp_var_res
        #  df_volatility = pd.DataFrame(var_res).T
        #  df_volatility = df_volatility / df_volatility.mean().mean()
        #  df_volatility.loc[df_inc.index[0]] = [1 for i in range(len(df_inc.columns))]
        #  df_volatility = df_volatility.sort_index()
        #  df_inc = df_inc / df_volatility
        #One month variation
        var = np.array([self.assets[code].origin_nav_sr.reindex(tdate).pct_change().loc[df_inc.index[-5]:df_inc.index[-1]].var() for code in df_inc.columns])
        risk, returns, ws, sharpe = PF.markowitz_bootstrape(df_inc, bound, cpu_count = self.cpu_count, bootstrap_count = self.bootstrap_count)
        ws = np.array(ws).ravel()
        ws = ws/var
        ws = ws/ws.sum()
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws




if __name__ == '__main__':

    #  asset = Asset('120000001')
    #  print asset.nav().tail()

    #  asset = WaveletAsset('120000013', 2)
    #  print asset.nav('1900-01-01', datetime.now()).tail()

    trade_date = ATradeDate.week_trade_date(begin_date = '2012-01-01')
    #  print trade_date[-5:]

    #  asset_globalids = ['120000001', '120000002', '120000013', '120000014', '120000015']
    #  assets = {}
    #  for asset_id in asset_globalids:
        #  assets[asset_id] = Asset(asset_id)
        #assets[asset_id] = WaveletAsset(asset_id, 2)

    #  allocate = AvgAllocate('ALC.000001', assets, trade_date, 14)
    #print allocate.allocate().tail()

    #  allocate = MzAllocate('ALC.000002', assets, trade_date, 14)
    #print allocate.allocate().tail()

    #  allocate = MzBootAllocate('ALC.000002', assets, trade_date, 14)
    #print allocate.allocate().tail()


    #  view_df = View.load_view('BL.000001')

    asset_globalids = ['120000001', '120000002', 'ERI000001', '120000014', 'ERI000002']
    assets = {}
    for asset_id in asset_globalids:
        assets[asset_id] = WaveletAsset(asset_id, 2)

    #  views = {}
    #  for asset_id in asset_globalids:
        #  views[asset_id] = View(None, asset_id, view_sr = view_df[asset_id], confidence = 0.5) if asset_id in view_df.columns else View(None, asset_id, confidence = 0.5)

    #  allocate = MzBootBlAllocate('ALC.000002', assets, views, trade_date, 26)
    #  print allocate.allocate().tail()

    allocate = MzVolAdjAllocate('ALC.000001', assets, trade_date, 14)
    print allocate.allocate().tail(10)