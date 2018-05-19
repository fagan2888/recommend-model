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
import copy
from util.xdebug import dd
from wavelet import Wavelet
from trade_date import ATradeDate
from scipy.stats import pearsonr


import traceback, code


logger = logging.getLogger(__name__)



class Asset(object):


    def __init__(self, globalid, name = None, nav_sr = None):

        self.__globalid = globalid
        self.__nav_sr = nav_sr
        self.__name = name


    @property
    def globalid(self):
        return self.__globalid


    @property
    def origin_nav_sr(self):
        if self.__nav_sr is None:
            self.__nav_sr = Asset.load_nav_series(self.__globalid)
        return self.__nav_sr.copy()


    @property
    def origin_nav_df(self):
        nav_df = self.origin_nav_sr.to_frame()
        nav_df.columns = [self.__globalid]
        return nav_df


    def nav(self, begin_date = None, end_date = None, reindex = None):

        nav_sr = self.origin_nav_sr

        if begin_date is not None:
            nav_sr = nav_sr[nav_sr.index >= begin_date]
        if end_date is not None:
            nav_sr = nav_sr[nav_sr.index <= end_date]
        if reindex is not None:
            nav_sr = nav_sr.reindex(reindex).fillna(method = 'pad')
            nav_sr = nav_sr.loc[reindex]

        return nav_sr


    @staticmethod
    def load_nav_series(asset_id, reindex=None, begin_date=None, end_date=None):

        prefix = asset_id[0:2]
        if prefix.isdigit():
            xtype = int(asset_id) / 10000000
            if xtype == 1:
                #
                # 基金池资产
                #
                asset_id = int(asset_id) % 10000000
                (pool_id, category) = (asset_id / 100, asset_id % 100)
                ttype = pool_id / 10000
                sr = asset_ra_pool_nav.load_series(
                    pool_id, category, ttype, reindex=reindex, begin_date=begin_date, end_date=end_date)
            elif xtype == 3:
                #
                # 基金池资产
                #
                sr = base_ra_fund_nav.load_series(
                    asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
            elif xtype == 4:
                #
                # 修型资产
                #
                sr = asset_rs_reshape_nav.load_series(
                    asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
            elif xtype == 12:
                #
                # 指数资产
                #
                sr = base_ra_index_nav.load_series(
                    asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
            else:
                sr = pd.Series()
        else:
            if prefix == 'AP':
                #
                # 基金池资产
                #
                sr = asset_ra_pool_nav.load_series(
                    asset_id, 0, 9, reindex=reindex, begin_date=begin_date, end_date=end_date)
            elif prefix == 'FD':
                #
                # 基金资产
                #
                sr = base_ra_fund_nav.load_series(
                    asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
            elif prefix == 'RS':
                #
                # 修型资产
                #
                sr = asset_rs_reshape_nav.load_series(
                    asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
            elif prefix == 'IX':
                #
                # 指数资产
                #
                sr = base_ra_index_nav.load_series(
                    asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
            elif prefix == 'ER':

                sr = base_exchange_rate_index_nav.load_series(
                    asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
            else:
                sr = pd.Series()

        return sr



class WaveletAsset(Asset):


    def __init__(self, globalid, wavelet_filter_num, wave_begin_date = None, name = None, nav_sr = None):

        super(WaveletAsset, self).__init__(globalid, name = name, nav_sr = nav_sr)
        self.__wavelet_filter_num = wavelet_filter_num
        self.__wave_begin_date = wave_begin_date
        if self.__wave_begin_date is None:
            self.__wave_begin_date = '1900-01-01'


    @property
    def wavelet_filter_num(self):
        return self.__wavelet_filter_num


    def nav(self, wave_end_date = None, begin_date = None, end_date = None, reindex = None):


        if reindex is not None:
            reindex.sort(reverse = False)
            begin_date = reindex[0]
            end_date = reindex[-1]
            wave_end_date = reindex[-1]
        elif end_date is not None:
            wave_end_date = end_date


        nav_sr = super(WaveletAsset, self).nav(begin_date = self.__wave_begin_date, end_date = wave_end_date)

        wavelet_nav_sr = Wavelet.wavefilter(nav_sr, self.wavelet_filter_num)


        if begin_date is not None:
            wavelet_nav_sr = wavelet_nav_sr[wavelet_nav_sr.index >= begin_date]
        if reindex is not None:
            wavelet_nav_sr = wavelet_nav_sr.reindex(reindex).fillna(method = 'pad')
            wavelet_nav_sr = wavelet_nav_sr.loc[reindex]

        return wavelet_nav_sr


class RecentStdAsset(Asset):

    def __init__(self, globalid, recent_period = 13, trade_dates = None, name = None, nav_sr = None):

        super(RecentStdAsset, self).__init__(globalid, name = name, nav_sr = nav_sr)
        self.__recent_period = recent_period
        self.__trade_dates = trade_dates

    @property
    def trade_dates(self):
        return self.__trade_dates.copy()

    @property
    def recent_period(self):
        return self.__recent_period


    def nav(self, begin_date = None, end_date = None, reindex = None):

        nav_sr = super(RecentStdAsset, self).nav()
        nav_sr = nav_sr.loc[self.trade_dates]

        if reindex is not None:
            reindex.sort(reverse = False)
            begin_date = reindex[0]
            end_date = reindex[-1]
        if begin_date is None:
            begin_date = self.trade_dates[0]
        if end_date is None:
            end_date = self.trade_dates[-1]

        recent_dates = self.recent_trade_dates(self.trade_dates, self.recent_period + 1, end_date)
        recent_df_nav = nav_sr.loc[recent_dates]
        recent_std = recent_df_nav.pct_change().dropna().std()

        dates = self.trade_dates
        dates = dates[dates >= begin_date]
        dates = dates[dates <= end_date]

        dates = list(dates)
        dates.sort()

        incs = []
        for day in dates:
            recent_dates = self.recent_trade_dates(self.trade_dates, self.recent_period + 1, day)
            inc = nav_sr.loc[recent_dates].pct_change().dropna()
            if len(inc) == self.recent_period:
                #print day, end_date, inc.std(), recent_std, self.globalid
                #print inc
                #print inc / inc.std() * recent_std
                #print
                inc = (inc / inc.std() * recent_std).ravel()[-1]
            else:
                inc = 0
            incs.append(inc)

        sr_inc = pd.Series(incs, index = dates)
        nav_sr = (1 + sr_inc).cumprod()

        return nav_sr


    def recent_trade_dates(self, trade_dates, recent_period, end_date):
        if end_date is not None:
            rtd = trade_dates[trade_dates <= end_date]
        else:
            rtd = trade_dates
        rtd = rtd[-1 * recent_period:]

        return rtd



class RollingAsset(Asset):


    def __init__(self, globalid, rolling = 20,  name = None, nav_sr = None):

        super(RollingAsset, self).__init__(globalid, name = name, nav_sr = nav_sr)
        self.__rolling = rolling

    @property
    def rolling(self):
        return self.__rolling

    def nav(self, begin_date = None, end_date = None, reindex = None):

        nav_sr = super(RollingAsset, self).nav()
        nav_sr = nav_sr.rolling(self.rolling).mean()

        if begin_date is not None:
            nav_sr = nav_sr[nav_sr.index >= begin_date]
        if reindex is not None:
            nav_sr = nav_sr.reindex(reindex).fillna(method = 'pad')
            nav_sr = nav_sr.loc[reindex]

        return nav_sr




class DoubleRollingAsset(Asset):


    def __init__(self, globalid, quick_rolling = 20, slow_rolling = 60, name = None, nav_sr = None):

        super(DoubleRollingAsset, self).__init__(globalid, name = name, nav_sr = nav_sr)
        self.__quick_rolling = quick_rolling
        self.__slow_rolling = slow_rolling

    @property
    def quick_rolling(self):
        return self.__quick_rolling

    @property
    def slow_rolling(self):
        return self.__slow_rolling

    def nav(self, begin_date = None, end_date = None, reindex = None):


        nav_sr = super(DoubleRollingAsset, self).nav()

        quick_nav_sr = nav_sr.rolling(self.quick_rolling).mean()
        slow_nav_sr = nav_sr.rolling(self.slow_rolling).mean()

        nav_sr = quick_nav_sr / slow_nav_sr - 1

        if begin_date is not None:
            nav_sr = nav_sr[nav_sr.index >= begin_date]
        if reindex is not None:
            nav_sr = nav_sr.reindex(reindex).fillna(method = 'pad')
            nav_sr = nav_sr.loc[reindex]

        #nav_sr = nav_sr.cumprod()

        return nav_sr

if __name__ == '__main__':

    asset = Asset('120000001')
    #print asset.nav(begin_date = '2010-01-01').head()
    #print asset.origin_nav_sr.head()

    asset = WaveletAsset('120000013', 2)
    #print asset.nav('2010-01-01', datetime.now()).tail()
    #print asset.origin_nav_sr.tail()

    week_trade_dates = ATradeDate.week_trade_date()
    asset = RecentStdAsset('120000001', trade_dates = week_trade_dates)



    for i in range(1, 80):
        asset = RollingAsset(str(120000000 + i), rolling = 5)
        #print asset.nav()
        #print asset.origin_nav_sr
        sr = asset.origin_nav_sr.pct_change().shift(-1)
        nav = asset.nav().pct_change()
        #nav[nav < 0.02] = 0
        df = pd.concat([sr, nav] ,axis = 1).dropna()
        print i, pearsonr(df.iloc[:,0], df.iloc[:,1])

    #for i in range(5, 60):
    #    for j in [20, 60, 120, 250]:
    #        asset = DoubleRollingAsset('120000015', quick_rolling = i, slow_rolling = j)
    #        sr = asset.origin_nav_sr.pct_change().shift(-1)
    #        nav = asset.nav()
    #        df = pd.concat([sr, nav] ,axis = 1).dropna()
    #        print i , j, pearsonr(df.iloc[:,0], df.iloc[:,1])


