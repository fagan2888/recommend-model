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
from heapq import nlargest

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
from db.asset_stock_factor import *
from util import xdict
from util.xdebug import dd
from asset import Asset, WaveletAsset
from allocate import Allocate
from trade_date import ATradeDate
from view import View
from RiskParity import cal_weight
import util_optimize
import RiskParity
from stock_factor import StockFactor
from multiprocessing import Pool
from scipy.stats import rankdata

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


class MzBootDownRiskAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None, cpu_count = None, bootstrap_count = 0):
        super(MzBootDownRiskAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        if cpu_count is None:
            count = multiprocessing.cpu_count() / 2
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


class FactorRpAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):

        super(FactorRpAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)

        self.sfe = load_stock_factor_exposure(stock_ids = assets.keys(), begin_date = '2012-01-01')
        self.sfr = load_stock_factor_return(begin_date = '2012-01-01')
        self.sfsr = load_stock_factor_specific_return(stock_ids = assets.keys(), begin_date = '2012-01-01')


    def allocate_algo(self, day, df_inc, bound):

        dates = df_inc.index
        begin_date = dates[0]
        end_date = dates[-1]
        # stock_ids = df_inc.columns.values

        # sfe = load_stock_factor_exposure(stock_ids = stock_ids, begin_date = begin_date, end_date = end_date)
        # sfr = load_stock_factor_return(begin_date = begin_date, end_date = end_date)
        # sfsr = load_stock_factor_specific_return(stock_ids = stock_ids, begin_date = begin_date, end_date = end_date)
        sfe = self.sfe.reset_index()
        sfr = self.sfr.reset_index()
        sfsr = self.sfsr.reset_index()

        sfe = sfe[sfe.trade_date >= begin_date][sfe.trade_date <= end_date]
        sfr = sfr[sfr.trade_date >= begin_date][sfr.trade_date <= end_date]
        sfsr = sfsr[sfsr.trade_date >= begin_date][sfsr.trade_date <= end_date]

        sfe = sfe.groupby(['stock_id', 'sf_id']).mean()
        sfe = sfe.unstack()
        sfe.columns = sfe.columns.droplevel(0)
        sfe = sfe.fillna(0.0)

        sfr = sfr.set_index(['trade_date', 'sf_id'])
        sfr = sfr.unstack()
        sfr.columns = sfr.columns.droplevel(0)

        sfsr = sfsr.set_index(['trade_date', 'stock_id'])
        sfsr = sfsr.unstack()
        sfsr.columns = sfsr.columns.droplevel(0)

        # V = df_inc.cov().values*1e4
        # factor_cov = sfr.cov().values*1e4
        V1 = np.dot(np.dot(sfe, sfr.corr()), sfe.T)
        V2 = np.multiply(np.eye(50), sfsr.cov().values)
        V = V1 + V2
        risk_budget = [1.0 / len(df_inc.columns)] * len(df_inc.columns)
        weight = cal_weight(V, risk_budget)
        print
        # print day, weight
        print(np.round(np.dot(weight, sfe), 3))
        ws = dict(zip(df_inc.columns.ravel(), weight))

        return ws


class FactorMzAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):

        super(FactorMzAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        sf_ids = ['SF.00000%d'%i for i in range(1, 9)]
        t = datetime.now()

        # print t
        # self.sfe = load_stock_factor_exposure(sf_ids = sf_ids, stock_ids = assets.keys(), begin_date = '2012-01-01')
        # self.sfe.to_csv('data/factor/stock_factor_exposure.csv', index_label = ['stock_id', 'sf_id', 'trade_date'])
        self.sfe = pd.read_csv('data/factor/stock_factor_exposure.csv', index_col = ['stock_id', 'sf_id', 'trade_date'], parse_dates = ['trade_date'])
        # print datetime.now() - t

        # self.sfr = load_stock_factor_return(sf_ids = sf_ids, begin_date = '2012-01-01')
        # self.sfr.to_csv('data/factor/stock_factor_return.csv', index_label = ['sf_id', 'trade_date'])
        self.sfr = pd.read_csv('data/factor/stock_factor_return.csv', index_col = ['sf_id', 'trade_date'], parse_dates = ['trade_date'])

        # self.sfsr = load_stock_factor_specific_return(stock_ids = assets.keys(), begin_date = '2012-01-01')
        # self.sfsr.to_csv('data/factor/stock_factor_specific_return.csv', index_label = ['stock_id', 'trade_date'])
        self.sfsr = pd.read_csv('data/factor/stock_factor_specific_return.csv', index_col = ['stock_id', 'trade_date'], parse_dates = ['trade_date'])


    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index = adjust_days, columns = asset_ids)

        t = datetime.now()
        df_inc_all, bound = self.load_all_data(adjust_days, asset_ids)
        print(datetime.now() - t)
        print(df_inc_all.index)
        print(df_inc_all.columns)

        s = 'perform %-12s' % self.__class__.__name__

        with open('data/result/win_ratio_22.csv', 'w') as f:
            f.write('date, result\n')

        with click.progressbar(
                adjust_days, label=s.ljust(30),
                item_show_func=lambda x:  x.strftime("%Y-%m-%d") if x else None) as bar:

            for day in bar:

                logger.debug("%s : %s", s, day.strftime("%Y-%m-%d"))
                df_inc = df_inc_all[df_inc_all.index <= day][-1 * self.lookback:]
                df_inc_2 = df_inc_all[df_inc_all.index > day][:self.lookback]

                ws = self.allocate_algo(day, df_inc, bound, df_inc_2)

                for asset_id in ws.keys():
                    pos_df.loc[day, asset_id] = ws[asset_id]

        return pos_df


    def allocate_algo(self, day, df_inc, bound, df_inc_2):

        print
        dates = df_inc.index
        begin_date = dates[0]
        end_date = dates[-1]
        # stock_ids = df_inc.columns.values

        # sfe = load_stock_factor_exposure(stock_ids = stock_ids, begin_date = begin_date, end_date = end_date)
        # sfr = load_stock_factor_return(begin_date = begin_date, end_date = end_date)
        # sfsr = load_stock_factor_specific_return(stock_ids = stock_ids, begin_date = begin_date, end_date = end_date)
        sfe = self.sfe.reset_index()
        sfr = self.sfr.reset_index()
        sfsr = self.sfsr.reset_index()

        sfe = sfe[sfe.trade_date >= begin_date][sfe.trade_date <= end_date]
        sfr = sfr[sfr.trade_date >= begin_date][sfr.trade_date <= end_date]
        sfsr = sfsr[sfsr.trade_date >= begin_date][sfsr.trade_date <= end_date]

        sfe = sfe.groupby(['stock_id', 'sf_id']).mean()
        sfe = sfe.unstack()
        sfe.columns = sfe.columns.droplevel(0)
        sfe = sfe.fillna(0.0)

        sfr = sfr.set_index(['trade_date', 'sf_id'])
        sfr = sfr.unstack()
        sfr.columns = sfr.columns.droplevel(0)

        sfsr = sfsr.set_index(['trade_date', 'stock_id'])
        sfsr = sfsr.unstack()
        sfsr.columns = sfsr.columns.droplevel(0)

        joint_stocks = sfe.index.intersection(sfsr.columns)
        sfe = sfe.loc[joint_stocks]
        sfsr = sfsr[joint_stocks]

        # M = np.dot(sfe, sfr.mean())
        V1 = np.dot(np.dot(sfe, sfr.corr()), sfe.T)
        V2 = np.diag(np.diag(sfsr.cov().values))
        V = V1 + V2

        df_inc = df_inc[joint_stocks]
        df_inc_2 = df_inc_2[joint_stocks]
        df_inc_2 = df_inc_2.fillna(0.0)
        V_simple = df_inc.cov().values
        V_real = df_inc_2.cov().values

        factor_error = np.power(np.sum(np.power(V - V_real, 2)), 0.5)
        base_error = np.power(np.sum(np.power(V - V_simple, 2)), 0.5)
        if factor_error < base_error:
            win = 1
        else:
            win = 0

        with open('data/result/win_ratio_22.csv', 'a+') as f:
            f.write('%s,%s\n'%(day, win))

        risk_budget = [1.0 / len(joint_stocks)] * len(joint_stocks)
        print(day, win, factor_error, base_error)

        # try:
        #     weight = cal_weight(V, risk_budget)
        # except Exception as e:
        #     print e
        #     weight = [np.nan] * len(joint_stocks)

        # try:
        #     risk, returns, weight, sharpe = PF.markowitz_r_spe_mv(M, V, bound)
        # except ValueError:
        #     print 'rank(A) < p'
        #     weight = [np.nan] * len(joint_stocks)
        # weight = np.array(weight).ravel()

        # print np.round(np.dot(weight, sfe), 3)
        # print weight
        # ws = dict(zip(df_inc.columns.ravel(), weight))

        weight = [np.nan] * len(joint_stocks)
        ws = dict(zip(joint_stocks, weight))

        return ws


class FactorValidAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):

        super(FactorValidAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        # sf_ids = ['SF.0000%02d'%i for i in range(1, 10)] + ['SF.1000%02d'%i for i in range(1, 29)]
        sf_ids = ['SF.0000%02d'%i for i in range(1, 10)]
        self.sf_ids = sf_ids

        # print t
        # self.sfe = load_stock_factor_exposure(sf_ids = sf_ids, stock_ids = assets.keys(), begin_date = '2010-01-01')
        # self.sfe.to_csv('data/factor/stock_factor_exposure.csv', index_label = ['stock_id', 'sf_id', 'trade_date'])
        self.sfe = pd.read_csv('data/factor/stock_factor_exposure.csv', index_col = ['stock_id', 'sf_id', 'trade_date'], parse_dates = ['trade_date'])
        # print datetime.now() - t

        # self.sfr = load_stock_factor_return(sf_ids = sf_ids, begin_date = '2012-01-01')
        # self.sfr.to_csv('data/factor/stock_factor_return.csv', index_label = ['sf_id', 'trade_date'])
        self.sfr = pd.read_csv('data/factor/stock_factor_return.csv', index_col = ['sf_id', 'trade_date'], parse_dates = ['trade_date'])
        self.sfr = self.sfr.unstack().T
        self.sfr.index = self.sfr.index.get_level_values(1)
        self.sfr = ((self.sfr - self.sfr.mean()) / self.sfr.std()) / 100
        self.sfr = (1+self.sfr).cumprod()
        self.sfr = self.sfr.rolling(self.lookback).mean().dropna()
        # sfr.to_csv('data/sfr.csv')

        # self.sfsr = load_stock_factor_specific_return(stock_ids = assets.keys(), begin_date = '2012-01-01')
        # self.sfsr.to_csv('data/factor/stock_factor_specific_return.csv', index_label = ['stock_id', 'trade_date'])
        self.sfsr = pd.read_csv('data/factor/stock_factor_specific_return.csv', index_col = ['stock_id', 'trade_date'], parse_dates = ['trade_date'])


    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index = adjust_days, columns = asset_ids)

        df_inc_all, bound = self.load_all_data(adjust_days, asset_ids)
        self.df_inc_all = df_inc_all.sort_index()

        # s = 'perform %-12s' % self.__class__.__name__
        # with click.progressbar(
        #         adjust_days, label=s.ljust(30),
        #         item_show_func=lambda x:  x.strftime("%Y-%m-%d") if x else None) as bar:

        #     for day in bar:

        #         logger.debug("%s : %s", s, day.strftime("%Y-%m-%d"))
        #         df_inc = df_inc_all[df_inc_all.index <= day][-1 * self.lookback:]

        #         ws = self.allocate_algo(day, df_inc, bound)

        #         for asset_id in ws.keys():
        #             pos_df.loc[day, asset_id] = ws[asset_id]
        headers = ['date,'] + ['SF.00000%d,'%i for i in range(1, 10)] + ['\n']
        with open('data/valid_factor.csv', 'w') as f:
            f.writelines(headers)

        pool = Pool(32)
        wss = pool.map(self.allocate_algo, adjust_days)
        pool.close()
        pool.join()
        # self.allocate_algo(adjust_days[0])


        for day, ws in zip(adjust_days, wss):
            for asset_id in ws.keys():
                pos_df.loc[day, asset_id] = ws[asset_id]

        return pos_df


    def allocate_algo(self, day):

        print()
        df_inc = self.df_inc_all[self.df_inc_all.index <= day][-1 * self.lookback:]

        dates = df_inc.index
        begin_date = dates.min()
        end_date = dates.max()

        sfe = self.sfe[(self.sfe.index.get_level_values(2) > begin_date) & (self.sfe.index.get_level_values(2) < end_date)].reset_index()
        # sfr = self.sfr[(self.sfr.index.get_level_values(1) > begin_date) & (self.sfr.index.get_level_values(1) < end_date)].reset_index()
        sfr = self.sfr[(self.sfr.index > begin_date) & (self.sfr.index < end_date)]
        sfr2 = self.sfr[(self.sfr.index > end_date - timedelta(365)) & (self.sfr.index < end_date)]
        # sfr2 = self.sfr[(self.sfr.index.get_level_values(1) > end_date - timedelta(365)) & (self.sfr.index.get_level_values(1) < end_date)].reset_index()
        sfsr = self.sfsr[(self.sfsr.index.get_level_values(1) > begin_date) & (self.sfsr.index.get_level_values(1) < end_date)].reset_index()

        sfe = sfe.groupby(['stock_id', 'sf_id']).mean()
        sfe = sfe.unstack()
        sfe.columns = sfe.columns.droplevel(0)
        sfe = sfe.dropna(how = 'all')
        sfe = sfe.fillna(0.0)
        # sfe = sfe.dropna()
        sfe = sfe[self.sf_ids]

        # sfr = sfr.set_index(['trade_date', 'sf_id'])
        # sfr = sfr.unstack()
        # sfr.columns = sfr.columns.droplevel(0)
        sfr = sfr[self.sf_ids]

        # sfr2 = sfr2.set_index(['trade_date', 'sf_id'])
        # sfr2 = sfr2.unstack()
        # sfr2.columns = sfr2.columns.droplevel(0)
        # sfr2 = sfr2[self.sf_ids]

        sfsr = sfsr.set_index(['trade_date', 'stock_id'])
        sfsr = sfsr.unstack()
        sfsr.columns = sfsr.columns.droplevel(0)

        joint_stocks = sfe.index.intersection(sfsr.columns)
        sfe = sfe.loc[joint_stocks]
        sfsr = sfsr[joint_stocks]

        # rf = 0.0
        # R = sfr.mean()
        R = sfr.iloc[-1] / sfr.iloc[0] - 1
        # C = sfr.cov()
        # risk_budget = [1.0 / len(sfr.columns)] * len(sfr.columns)
        C = sfr2.cov()
        # factor_weights = util_optimize.mv_weights(R, C, rf)

        # factor_weights = RiskParity.cal_weight_factor(C, risk_budget)
        # factor_weights = np.sign(factor_weights) * 0.5

        factor_weights = np.sign(R.values)
        # factor_weights = (rankdata(np.diag(C)) / 10) * np.sign(R.values)
        # thresh = nlargest(5, abs(R.values))[-1]
        # valid_factors = np.where(abs(R.values) >= thresh, 1, 0)
        # valid_factors = rankdata(np.diag(C)) / 10
        print(day, factor_weights)

        factor_weights_str = [str(x) for x in factor_weights]
        with open('data/valid_factor.csv', 'a+') as f:
            f.write('%s,'%end_date + ','.join(factor_weights_str) + '\n')

        # factor_weights = R.values / R.values.std()
        print(end_date, np.round(sum(factor_weights), 2), np.round(factor_weights, 2))
        stock_weights = PureFactor.cal_weight(sfe, factor_weights)
        # stock_weights = PureFactor.cal_weight(sfe, factor_weights, x_v = valid_factors)
        # stock_weights = [0.0] * len(joint_stocks)
        print('total weight:', np.sum(stock_weights))
        # portfolio_exposure = np.dot(stock_weights, sfe)
        # print('exposure difference', np.sum(np.abs(factor_weights - portfolio_exposure)))

        ws = dict(zip(joint_stocks, stock_weights))

        return ws


class FactorSizeAllocate(Allocate):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None, target = None):

        super(FactorSizeAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        self.target = target
        sf_ids = ['SF.00000%d'%i for i in range(1, 10)]
        t = datetime.now()

        print(t)
        self.sfe = load_stock_factor_exposure(sf_ids = sf_ids, stock_ids = assets.keys(), begin_date = '2012-01-01')
        self.sfe.to_csv('data/factor/stock_factor_exposure.csv', index_label = ['stock_id', 'sf_id', 'trade_date'])
        # self.sfe = pd.read_csv('data/factor/stock_factor_exposure.csv', index_col = ['stock_id', 'sf_id', 'trade_date'], parse_dates = ['trade_date'])
        print('load exposure:', datetime.now() - t)
        # os._exit(0)

        self.sfr = load_stock_factor_return(sf_ids = sf_ids, begin_date = '2012-01-01')
        self.sfr.to_csv('data/factor/stock_factor_return.csv', index_label = ['sf_id', 'trade_date'])
        # self.sfr = pd.read_csv('data/factor/stock_factor_return.csv', index_col = ['sf_id', 'trade_date'], parse_dates = ['trade_date'])

        self.sfsr = load_stock_factor_specific_return(stock_ids = assets.keys(), begin_date = '2012-01-01')
        self.sfsr.to_csv('data/factor/stock_factor_specific_return.csv', index_label = ['stock_id', 'trade_date'])
        # self.sfsr = pd.read_csv('data/factor/stock_factor_specific_return.csv', index_col = ['stock_id', 'trade_date'], parse_dates = ['trade_date'])

        sfr = self.sfr.unstack().T
        sfr = (1+sfr).cumprod()
        sfr.to_csv('data/sfr.csv')
        os._exit(0)


    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index = adjust_days, columns = asset_ids)

        df_inc_all, bound = self.load_all_data(adjust_days, asset_ids)

        s = 'perform %-12s' % self.__class__.__name__

        with click.progressbar(
                adjust_days, label=s.ljust(30),
                item_show_func=lambda x:  x.strftime("%Y-%m-%d") if x else None) as bar:

            for day in bar:

                logger.debug("%s : %s", s, day.strftime("%Y-%m-%d"))
                df_inc = df_inc_all[df_inc_all.index <= day][-1 * self.lookback:]
                df_inc_2 = df_inc_all[df_inc_all.index > day][:self.lookback]

                ws = self.allocate_algo(day, df_inc, bound, df_inc_2)

                for asset_id in ws.keys():
                    pos_df.loc[day, asset_id] = ws[asset_id]

        return pos_df


    def allocate_algo(self, day, df_inc, bound, df_inc_2):

        print
        dates = df_inc.index
        begin_date = dates[0]
        end_date = dates[-1]
        # stock_ids = df_inc.columns.values

        # sfe = load_stock_factor_exposure(stock_ids = stock_ids, begin_date = begin_date, end_date = end_date)
        # sfr = load_stock_factor_return(begin_date = begin_date, end_date = end_date)
        # sfsr = load_stock_factor_specific_return(stock_ids = stock_ids, begin_date = begin_date, end_date = end_date)
        '''
        set_trace()
        sfe = self.sfe.reset_index()
        sfr = self.sfr.reset_index()
        sfsr = self.sfsr.reset_index()
        t2 = datetime.now()

        sfe = sfe[sfe.trade_date >= begin_date][sfe.trade_date <= end_date]
        sfr = sfr[sfr.trade_date >= begin_date][sfr.trade_date <= end_date]
        sfsr = sfsr[sfsr.trade_date >= begin_date][sfsr.trade_date <= end_date]
        t3 = datetime.now()
        '''
        sfe = self.sfe[(self.sfe.index.get_level_values(2) > begin_date) & (self.sfe.index.get_level_values(2) < end_date)].reset_index()
        sfr = self.sfr[(self.sfr.index.get_level_values(1) > begin_date) & (self.sfr.index.get_level_values(1) < end_date)].reset_index()
        sfsr = self.sfsr[(self.sfsr.index.get_level_values(1) > begin_date) & (self.sfsr.index.get_level_values(1) < end_date)].reset_index()

        sfe = sfe.groupby(['stock_id', 'sf_id']).mean()
        sfe = sfe.unstack()
        sfe.columns = sfe.columns.droplevel(0)
        sfe = sfe.dropna(how = 'all')
        sfe = sfe.fillna(0.0)

        sfr = sfr.set_index(['trade_date', 'sf_id'])
        sfr = sfr.unstack()
        sfr.columns = sfr.columns.droplevel(0)

        sfsr = sfsr.set_index(['trade_date', 'stock_id'])
        sfsr = sfsr.unstack()
        sfsr.columns = sfsr.columns.droplevel(0)

        joint_stocks = sfe.index.intersection(sfsr.columns)
        sfe = sfe.loc[joint_stocks]
        sfsr = sfsr[joint_stocks]

        # target = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        weight = PureFactor.cal_weight(sfe, self.target)
        print('max weight:', max(weight))
        print(day, np.dot(weight, sfe))

        # weight = [np.nan] * len(joint_stocks)
        ws = dict(zip(joint_stocks, weight))

        return ws


class FactorIndexAllocate(Allocate):

    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None, target = None):

        super(FactorIndexAllocate, self).__init__(globalid, assets, reindex, lookback, period, bound)
        # sf_ids = ['SF.0000%02d'%i for i in range(1, 10)] + ['SF.1000%02d'%i for i in range(1, 29)]
        sf_ids = ['SF.0000%02d'%i for i in range(1, 10)]
        self.sf_ids = sf_ids

        if target is None:
            self.target = [1] + [0] * 8
        else:
            self.target = target

        # self.sfe = load_stock_factor_exposure(sf_ids = sf_ids, stock_ids = assets.keys(), begin_date = '2010-01-01')
        # self.sfe.to_csv('data/factor/stock_factor_exposure.csv', index_label = ['stock_id', 'sf_id', 'trade_date'])
        self.sfe = pd.read_csv('data/factor/stock_factor_exposure.csv', index_col = ['stock_id', 'sf_id', 'trade_date'], parse_dates = ['trade_date'])

    def allocate(self):

        adjust_days = self.index[self.lookback - 1::self.period]
        asset_ids = list(self.assets.keys())
        pos_df = pd.DataFrame(0, index = adjust_days, columns = asset_ids)

        df_inc_all, bound = self.load_all_data(adjust_days, asset_ids)
        self.df_inc_all = df_inc_all.sort_index()

        pool = Pool(32)
        wss = pool.map(self.allocate_algo, adjust_days)
        pool.close()
        pool.join()

        for day, ws in zip(adjust_days, wss):
            for asset_id in ws.keys():
                pos_df.loc[day, asset_id] = ws[asset_id]

        return pos_df

    def allocate_algo(self, day):

        print(day)
        df_inc = self.df_inc_all[self.df_inc_all.index <= day][-1 * self.lookback:]

        dates = df_inc.index
        begin_date = dates.min()
        end_date = dates.max()

        sfe = self.sfe[(self.sfe.index.get_level_values(2) > begin_date) & (self.sfe.index.get_level_values(2) < end_date)].reset_index()

        sfe = sfe.groupby(['stock_id', 'sf_id']).mean()
        sfe = sfe.unstack()
        sfe.columns = sfe.columns.droplevel(0)
        sfe = sfe.dropna(how = 'all')
        sfe = sfe.fillna(0.0)
        sfe = sfe[self.sf_ids]

        stock_weights = IndexFactor.cal_weight(sfe, self.target)

        ws = dict(zip(sfe.index, stock_weights))

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
