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
from db import asset_ra_pool, asset_ra_pool_nav, asset_rs_reshape, asset_rs_reshape_nav, asset_rs_reshape_pos
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates, base_exchange_rate_index_nav, asset_ra_bl, asset_stock
from db.asset_stock_factor import *
from util import xdict
from util.xdebug import dd
from asset import Asset, WaveletAsset
from allocate import Allocate
from trade_date import ATradeDate
from view import View
from RiskParity import cal_weight
import util_optimize
from multiprocessing import Pool

import PureFactor
import IndexFactor
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


class FactorValidAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):

        super(FactorValidAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        sf_ids = ['SF.0000%02d'%i for i in range(1, 10)]
        self.sf_ids = sf_ids

        self.sfe = load_stock_factor_exposure(sf_ids = sf_ids, stock_ids = assets.keys(), begin_date = '2010-01-01')
        self.sfe = pd.read_csv('data/factor/stock_factor_exposure.csv', index_col = ['stock_id', 'sf_id', 'trade_date'], parse_dates = ['trade_date'])
        self.sfr = load_stock_factor_return(sf_ids = sf_ids, begin_date = '2012-01-01')

    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index = adjust_days, columns = asset_ids)

        pool = Pool(32)
        wss = pool.map(self.allocate_algo, adjust_days)
        pool.close()
        pool.join()

        for day, ws in zip(adjust_days, wss):
            for asset_id in ws.keys():
                pos_df.loc[day, asset_id] = ws[asset_id]

        return pos_df


    def allocate_algo(self, day):

        begin_date = (day.date() - timedelta(self.lookback - 1)).strftime('%Y-%m-%d')
        end_date = day.date().strftime('%Y-%m-%d')

        sfe = self.sfe[(self.sfe.index.get_level_values(2) > begin_date) & (self.sfe.index.get_level_values(2) < end_date)].reset_index()
        sfr = self.sfr[(self.sfr.index.get_level_values(1) > begin_date) & (self.sfr.index.get_level_values(1) < end_date)].reset_index()

        sfe = sfe.groupby(['stock_id', 'sf_id']).mean()
        sfe = sfe.unstack()
        sfe.columns = sfe.columns.droplevel(0)
        sfe = sfe.dropna(how = 'all')
        sfe = sfe.fillna(0.0)
        sfe = sfe[self.sf_ids]

        sfr = sfr.set_index(['trade_date', 'sf_id'])
        sfr = sfr.unstack()
        sfr.columns = sfr.columns.droplevel(0)
        sfr = sfr[self.sf_ids]

        R = sfr.mean()
        factor_weights = np.sign(R.values) * 0.5

        stock_weights = PureFactor.cal_weight(sfe, factor_weights)
        ws = dict(zip(sfe.index, stock_weights))

        return ws


class FactorIndexAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None, target = None):

        super(FactorIndexAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        sf_ids = ['SF.0000%02d'%i for i in range(1, 10)]
        self.sf_ids = sf_ids

        if target is None:
            self.target = [1] + [0] * 8
        else:
            self.target = target


        self.sfe = load_stock_factor_exposure(sf_ids = sf_ids, begin_date = '2010-01-01')
        # self.sfe.to_csv('data/factor/stock_factor_exposure.csv', index_label = ['stock_id', 'sf_id', 'trade_date'])
        # self.sfe = pd.read_csv('data/factor/stock_factor_exposure.csv', index_col = ['stock_id', 'sf_id', 'trade_date'], parse_dates = ['trade_date'])

    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]

        df = pd.DataFrame()
        pos_df = {}

        pool = Pool(16)
        wss = pool.map(self.allocate_algo, adjust_days)
        pool.close()
        pool.join()
        # self.allocate_algo(adjust_days[0])

        for day, ws in zip(adjust_days, wss):
            pos_df[day] = ws

        pos_df = df.from_dict(pos_df, orient = 'index')
        pos_df = pos_df.fillna(0.0)

        return pos_df

    def allocate_algo(self, day):
        print(day)

        index_pos = asset_stock.load_index_pos('2070000191', day)
        asset_ids = self.sfe.index.levels[0].intersection(index_pos).values
        sfe = self.sfe.loc[asset_ids]

        begin_date = (day.date() - timedelta(self.lookback - 1)).strftime('%Y-%m-%d')
        end_date = day.date().strftime('%Y-%m-%d')

        sfe = sfe[(sfe.index.get_level_values(2) > begin_date) & (sfe.index.get_level_values(2) < end_date)]
        sfe = sfe.reset_index()
        sfe = sfe.groupby(['stock_id', 'sf_id']).mean()
        sfe = sfe.unstack()
        sfe.columns = sfe.columns.droplevel(0)
        sfe = sfe.dropna(how = 'all')
        sfe = sfe.fillna(0.0)
        sfe = sfe[self.sf_ids]

        stock_weights = IndexFactor.cal_weight(sfe, self.target)

        ws = dict(zip(sfe.index, stock_weights))

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
