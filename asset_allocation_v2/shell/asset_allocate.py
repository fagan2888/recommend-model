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
import Financial as fin
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
from allocate import Allocate, AssetBound
from trade_date import ATradeDate
from view import View
import RiskParity
import util_optimize
from multiprocessing import Pool

import PureFactor
import IndexFactor
import traceback, code
from monetary_fund_filter import MonetaryFundFilter


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


class RpAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):
        super(RpAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)


    def allocate_algo(self, day, df_inc, bound):
        asset_num = df_inc.shape[1]
        V = df_inc.cov()
        x_t = np.array([1 / asset_num] * asset_num)
        ws = RiskParity.cal_weight(V, x_t)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws



class MzBlAllocate(Allocate):


    def __init__(self, globalid, assets, views, reindex, lookback, period = 1, bound = None):
        super(MzBlAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        self.__views = views

    def allocate_algo(self, day, df_inc, bound):
        P, eta, alpha = self.load_bl_view(day, df_inc.columns)
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
        #print(ws)
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
        P, eta, alpha = self.load_bl_view(day, df_inc.columns)
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
        wavelet_asset_ids = ['120000001','120000002', '120000010', '120000011' ,'120000016', '120000080', '120000081', '120000082', 'ERI000001', 'ERI000002']
        wavelet_df_inc, wavelet_bound = self.load_wavelet_allocate_data(day, wavelet_asset_ids)
        wavelet_df_inc[wavelet_df_inc > 0] = 0.0
        risk, returns, ws, sharpe = PF.markowitz_r_spe(wavelet_df_inc, wavelet_bound)
        #risk, returns, ws, sharpe = PF.markowitz_bootstrape(wavelet_df_inc, wavelet_bound, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        wavelet_ws = dict(zip(wavelet_df_inc.columns.ravel(), ws))
        cols = []
        for asset_id in df_inc.columns:
            if asset_id in set(['120000039']):
                cols.append(asset_id)
        if len(cols) == 0:
            return wavelet_ws

        wavelet_inc = pd.Series(np.zeros(len(df_inc)), index = df_inc.index)
        for asset_id in wavelet_ws:
            pos = wavelet_ws[asset_id]
            wavelet_inc = wavelet_inc + df_inc[asset_id] * pos

        fix_risk_asset_inc = df_inc[cols]
        fix_risk_asset_inc['wavelet_inc'] = wavelet_inc
        fix_risk_asset_bound = []
        for i in range(0, len(fix_risk_asset_inc.columns)):
            asset_bound = bound[i]
            asset_bound['upper'] = 1.0
            fix_risk_asset_bound.append(asset_bound)

        risk, returns, fix_risk_asset_ws, sharpe = PF.markowitz_bootstrape_fixrisk(fix_risk_asset_inc, fix_risk_asset_bound, self.risk, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        fix_risk_asset_ws = dict(zip(fix_risk_asset_inc.columns.ravel(), fix_risk_asset_ws))
        if fix_risk_asset_ws['120000039'] < 0.25:
            fix_risk_asset_ws['wavelet_inc'] = fix_risk_asset_ws['wavelet_inc'] + fix_risk_asset_ws['120000039']
            fix_risk_asset_ws['120000039'] = 0
        final_ws = {}
        for key in fix_risk_asset_ws.keys():
            if key == 'wavelet_inc':
                for wavelet_asset_id in wavelet_ws.keys():
                    final_ws[wavelet_asset_id] = wavelet_ws[wavelet_asset_id] * fix_risk_asset_ws[key]
            else:
                final_ws[key] = fix_risk_asset_ws[key]

        return final_ws


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



    def load_wavelet_allocate_data(self, day ,asset_ids):

        reindex = self.index[self.index <= day][-1 * self.lookback:]

        bound = []
        allocate_asset_ids = []
        for asset_id in asset_ids:
            asset_bound = AssetBound.get_asset_day_bound(asset_id, day, self.bound).to_dict()
            if asset_bound['upper'] > 0:
                bound.append(asset_bound)
                allocate_asset_ids.append(asset_id)

        data = {}
        for asset_id in allocate_asset_ids:
            data[asset_id] = self.wavelet_assets[asset_id].nav(reindex = reindex)
        df_nav = pd.DataFrame(data).fillna(method='pad')
        df_inc  = df_nav.pct_change().fillna(0.0)
        #df_inc  = np.log(df_nav).diff().fillna(0.0)

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
        P, eta, alpha = self.load_bl_view(day, df_inc.columns)
        risk, returns, ws, sharpe = PF.markowitz_bootstrape_bl_fixrisk(df_inc, P, eta, alpha ,bound, self.risk, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        ws = dict(zip(df_inc.columns.ravel(), ws))
        return ws


    def load_wavelet_allocate_data(self, day ,asset_ids):

        reindex = self.index[self.index <= day][-1 * self.lookback:]

        bound = []
        allocate_asset_ids = []
        for asset_id in asset_ids:
            asset_bound = AssetBound.get_asset_day_bound(asset_id, day, self.bound).to_dict()
            if asset_bound['upper'] > 0:
                bound.append(asset_bound)
                allocate_asset_ids.append(asset_id)

        data = {}
        for asset_id in allocate_asset_ids:
            data[asset_id] = self.wavelet_assets[asset_id].nav(reindex = reindex)
        df_nav = pd.DataFrame(data).fillna(method='pad')
        df_inc  = df_nav.pct_change().fillna(0.0)

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

        P, eta, alpha = self.load_bl_view(day, df_inc.columns)
        risk, returns, ws, sharpe = PF.markowitz_bootstrape_bl_fixrisk(df_inc, P, eta, alpha, bound, self.risk, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        ws = dict(zip(df_inc.columns.ravel(), ws))

        return ws


class FactorValidAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None, cpu_count = None):

        super(FactorValidAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        sf_ids = ['SF.0000%02d'%i for i in range(1, 10)]
        self.sf_ids = sf_ids

        self.sfe = load_stock_factor_exposure(sf_ids = sf_ids, stock_ids = assets.keys(), begin_date = '2010-01-01')
        self.sfr = load_stock_factor_return(sf_ids = sf_ids, begin_date = '2012-01-01')

        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self.__cpu_count = cpu_count
        else:
            self.__cpu_count = cpu_count

    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index = adjust_days, columns = asset_ids)

        pool = Pool(self.__cpu_count)
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

    def __init__(self, globalid, reindex, lookback, assets = None, period = 1, bound = None, target = None):

        super(FactorIndexAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        sf_ids = ['SF.0000%02d'%i for i in range(1, 10)]
        self.sf_ids = sf_ids

        if target is None:
            self.target = [1] + [0] * 8
        else:
            self.target = target


        self.sfe = load_stock_factor_exposure(sf_ids = sf_ids, begin_date = '2007-10-01')

    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]

        df = pd.DataFrame()
        pos_df = {}

        pool = Pool(8)
        wss = pool.map(self.allocate_algo, adjust_days)
        pool.close()
        pool.join()

        # wss = []
        # for day in adjust_days:
        #     wss.append(self.allocate_algo(day))

        for day, ws in zip(adjust_days, wss):
            pos_df[day] = ws

        pos_df = df.from_dict(pos_df, orient = 'index')
        pos_df = pos_df.fillna(0.0)

        return pos_df

    def allocate_algo(self, day):

        index_pos = asset_stock.load_index_pos('2070000191', day)
        asset_ids = self.sfe.index.levels[0].intersection(index_pos).values

        begin_date = (day.date() - timedelta(self.lookback - 1)).strftime('%Y-%m-%d')
        end_date = day.date().strftime('%Y-%m-%d')

        sfe = self.sfe[(self.sfe.index.get_level_values(2) > begin_date) & (self.sfe.index.get_level_values(2) < end_date)]
        sfe = sfe.loc[asset_ids]
        sfe = sfe.reset_index()
        sfe = sfe.groupby(['stock_id', 'sf_id']).mean()
        sfe = sfe.unstack()
        sfe.columns = sfe.columns.droplevel(0)
        sfe = sfe.dropna(how = 'all')
        sfe = sfe.fillna(0.0)
        print(day)
        sfe = sfe[self.sf_ids]

        stock_weights = IndexFactor.cal_weight(sfe, self.target)

        ws = dict(zip(sfe.index, stock_weights))

        return ws


class MzLayerFixRiskBootBlAllocate(MzBlAllocate):

    def __init__(self, globalid, assets, views, reindex, lookback, risk, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzLayerFixRiskBootBlAllocate, self).__init__(globalid, assets, views, reindex, lookback, period, bound)
        if cpu_count is None:
            count = int(multiprocessing.cpu_count()) // 2
            cpu_count = count if count > 0 else 1
            self.__cpu_count = cpu_count
        else:
            self.__cpu_count = cpu_count
        self.__bootstrap_count = bootstrap_count
        self.risk = risk

    def allocate_algo(self, day, df_inc, bound):

        all_assets = {}
        all_assets['stock'] = ['120000001', '120000002','120000018','ERI000001', 'ERI000002', '120000082']
        all_assets['bond'] = ['120000010', '120000011']
        all_assets['money'] = ['120000039']
        all_assets['gold'] = ['120000014']
        all_assets['oil'] = ['120000081']

        all_asset_bound = []
        _bound = bound[0].copy()
        _bound['upper'] = 1.0
        all_asset_bound.append(_bound)
        _bound = bound[0].copy()
        _bound['upper'] = 1.0
        all_asset_bound.append(_bound)
        _bound = bound[0].copy()
        _bound['upper'] = 1.0
        all_asset_bound.append(_bound)
        _bound = bound[0].copy()
        _bound['upper'] = 0.8
        all_asset_bound.append(_bound)
        _bound = bound[0].copy()
        _bound['upper'] = 0.1
        all_asset_bound.append(_bound)

        all_asset_nav = {}
        all_asset_ws = {}
        for key in all_assets.keys():
            asset_ids = all_assets[key]
            if len(asset_ids) > 1:
                df_inc_layer = df_inc[asset_ids]
                layer_bound = []
                for asset_id in asset_ids:
                    for _bound in bound:
                        if _bound['asset_id'] == asset_id:
                            layer_bound.append(_bound)
                            break
                risk, returns, ws, sharpe = PF.markowitz_bootstrape(df_inc_layer, layer_bound, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
                all_asset_nav[key] = (df_inc_layer * ws).sum(axis = 1)
                all_asset_ws[key] = ws
            else:
                all_asset_nav[key] = df_inc[asset_ids[0]]
                all_asset_ws[key] = [1.0]

        all_asset_df_inc = pd.DataFrame(all_asset_nav)

        risk, returns, ws, sharpe = PF.markowitz_bootstrape_fixrisk(all_asset_df_inc, all_asset_bound, self.risk, cpu_count = self.__cpu_count, bootstrap_count = self.__bootstrap_count)
        ws = dict(zip(all_asset_df_inc.columns.ravel(), ws))

        all_ws = {}
        for key in ws.keys():
            w = ws[key]
            layer_asset_ws = all_asset_ws[key]
            for i in range(0, len(all_assets[key])):
                asset_id = all_assets[key][i]
                all_ws[asset_id] = layer_asset_ws[i] * w

        return all_ws


class SingleValidFactorAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, alloc_num, bound=None, period=1):
        super(SingleValidFactorAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        self.alloc_num = alloc_num

    def allocate_algo(self, day, df_inc, bound):

        df_mean = df_inc.mean()
        df_pos = df_mean.copy()
        valid_factors = df_mean.nlargest(self.alloc_num).index
        invalid_factors = df_mean.index.difference(valid_factors)
        df_pos.loc[valid_factors] = 1.0 / self.alloc_num
        df_pos.loc[invalid_factors] = 0.0
        ws = df_pos.to_dict()

        return ws


class MonetaryAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period=1, bound=None, alloc_num=2):
        super(MonetaryAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        mnf = MonetaryFundFilter()
        mnf.handle()
        self.mnf = mnf
        self.alloc_num = alloc_num

    def allocate_algo(self, day, df_inc, bound):

        fund_status = self.mnf.fund_status
        fund_status = fund_status[fund_status.fi_yingmi_amount <= 1e3]
        fund_status = fund_status[fund_status.fi_yingmi_subscribe_status == 0.0]
        valid_ids_1 = fund_status.index
        fund_fee = self.mnf.fund_fee.ff_fee
        valid_ids_2 = fund_fee[fund_fee >= 0.2].index
        valid_ids_2 = [str(fund_code) for fund_code in valid_ids_2]
        valid_ids = np.intersect1d(valid_ids_1, valid_ids_2)

        tmp_scale = self.mnf.fund_scale.loc[day]
        tmp_scale = tmp_scale.sort_values(ascending=False)
        scale_filter_codes = tmp_scale[tmp_scale > 1e10].index
        scale_filter_ids = [str(self.mnf.fund_id_dict[fund_code]) for fund_code in scale_filter_codes]

        final_filter_ids = np.intersect1d(scale_filter_ids, valid_ids)
        tmp_df_inc = df_inc.copy()
        tmp_df_inc = tmp_df_inc[final_filter_ids]

        ws = {}
        num = self.alloc_num
        rs = tmp_df_inc.mean()
        rs = rs.sort_values(ascending=False)
        num = min(num, len(rs))
        for i in range(0, num):
            fund_globalid = rs.index[i]
            ws[fund_globalid] = 1.0 / num
            # print(fund_globalid, fund_fee.loc[int(fund_globalid)])

        # print(ws)
        return ws

    @staticmethod
    def all_monetary_fund_globalid():
        all_monetary_fund_df = base_ra_fund.find_type_fund(3)
        all_monetary_fund_df = all_monetary_fund_df.set_index(['globalid'])
        return all_monetary_fund_df.index.astype(str).tolist()


class CppiAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, period=1, bound=None, forcast_days=90, var_percent=10):
        super(CppiAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        self.forcast_days = forcast_days
        self.var_percent = var_percent

    def allocate_cppi(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        sdate = adjust_days[0].date()

        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(columns=asset_ids)

        # allocate monetary fund for first 3 years
        edate_3m = sdate + timedelta(90)
        edate_3m = datetime(edate_3m.year, edate_3m.month, edate_3m.day)
        pre_3m = adjust_days[adjust_days <= edate_3m]
        adjust_days = adjust_days[adjust_days > edate_3m]
        pos_df = pd.DataFrame(columns=asset_ids, index=pre_3m)
        pos_df = pos_df.fillna(0.0)
        pos_df['120000039'] = 1.0

        self.pos_df = pos_df
        for day in adjust_days:

            df_inc, bound = self.load_allocate_data(day, asset_ids)
            ws = self.allocate_algo(day, df_inc, bound)

            for asset_id in list(ws.keys()):
                pos_df.loc[day, asset_id] = ws[asset_id]
            self.pos_df = pos_df

        return pos_df

    def allocate_algo(self, day, df_inc, bound):

        bond_var = CppiAllocate.bond_var(self.forcast_days, self.var_percent, day)
        monetary_ret = 1.04**(self.forcast_days / 365) - 1

        df_pos = self.pos_df.copy()
        df_nav_portfolio = DFUtil.portfolio_nav(df_inc, df_pos, result_col='portfolio')
        df_nav = df_nav_portfolio.portfolio

        tmp_ret = df_nav.iloc[-1] / df_nav.iloc[0]
        tmp_benchmark_ret = 0
        tmp_overret = tmp_ret - tmp_benchmark_ret

        tmp_dd = 1 - df_nav.iloc[-1] / df_nav.max()
        money_pos = df_pos.iloc[-1].loc['120000039']
        # print(tmp_dd)

        if (tmp_dd > 0.01) or (money_pos == 1.0 and tmp_dd > 0):
            ws = CppiAllocate.money_alloc()
        elif tmp_overret <= 0:
            ws = CppiAllocate.money_alloc()
        else:
            # 未来3个月有90%的概率战胜净值不跌到余额宝之下
            lr = 1 / (monetary_ret - bond_var)
            ws = CppiAllocate.cppi_alloc(tmp_overret, tmp_ret, lr)
        # print(day, ws)
        # print(day, tmp_overret, tmp_ret)

        return ws

    @staticmethod
    def money_alloc():

        ws = {}
        ws['120000010'] = 0.0
        ws['120000011'] = 0.0
        ws['120000039'] = 1.0

        return ws

    @staticmethod
    def cppi_alloc(overret, tnav, lr):

        ws = {}
        ws_b = (overret - 1) / tnav / 2 * lr
        ws_m = 1 - ws_b * 2
        if ws_m > 0.25:
            ws['120000010'] = ws_b
            ws['120000011'] = ws_b
            ws['120000039'] = ws_m
        else:
            ws['120000010'] = 0.375
            ws['120000011'] = 0.375
            ws['120000039'] = 0.25

        return ws

    @staticmethod
    def bond_var(period, percent, day):
        df_nav = base_ra_index_nav.load_series('120000010')
        df_inc = df_nav.pct_change(period)
        df_inc = df_inc.dropna()
        df_inc = df_inc[df_inc.index < day]
        df_inc = df_inc.iloc[-365*5:]
        var = np.percentile(df_inc, percent)
        return var



if __name__ == '__main__':


    print(MonetaryAllocate.all_monetary_fund_globalid())

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
