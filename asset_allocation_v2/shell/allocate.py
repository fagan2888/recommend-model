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
from trade_date import ATradeDate


import traceback, code


logger = logging.getLogger(__name__)


class AssetBound(object):

    def __init__(self, globalid = None, asset_ids = None, bound = None, upper = 0.7):

        self.__globalid = globalid
        #TODO : load from database by globalid

        self.__asset_ids = asset_ids

        if bound is None:

            asset_bound = {'sum1': 0, 'sum2' : 0, 'upper': upper, 'lower': 0.0, 'lower_sum1' : 0, 'lower_sum2' : 0, 'upper_sum1' : 0, 'upper_sum2' : 0}
            trade_date = [pd.datetime(1900,1,1)]
            date_num = len(trade_date)
            trade_date = trade_date * len(asset_ids)
            trade_date.sort(reverse = False)
            asset_ids = list(asset_ids) * date_num
            date_asset_index = pd.MultiIndex.from_arrays([trade_date, asset_ids], names = ['trade_date', 'asset'])
            self.__bound = pd.DataFrame([asset_bound] * len(date_asset_index), index = date_asset_index)

        else:

            self.__bound = bound

    @property
    def globalid(self):
        return self.__globalid

    @property
    def asset_ids(self):
        return self.__asset_ids.copy()

    @property
    def bound(self):
        return self.__bound.copy()

    def get_day_bound(self, day):
        bound = self.__bound.copy().sort_index(ascending=True)
        days = bound.index.get_level_values(0)
        last_day = days[-1]
        return bound.loc[last_day]

    def load_db():
        pass


    def to_db():
        pass



class Allocate(object):


    def __init__(self, globalid, assets, reindex, lookback, period = 1, bound = None):

        self.__globalid = globalid
        self.__assets = assets
        self.__pos = None
        self.__nav_sr = None
        self.__index = reindex
        self.__lookback = lookback
        self.__period = period

        if bound is None:
            self.__bound = AssetBound('asset_bound_default', [asset_id for asset_id in list(assets.keys())])
        else:
            self.__bound = bound

    @property
    def globalid(self):
        return self.__globalid

    @property
    def assets(self):
        return self.__assets

    @property
    def index(self):
        return self.__index

    @property
    def lookback(self):
        return self.__lookback

    @property
    def period(self):
        return self.__period

    @property
    def nav_sr(self):
        return self.__nav_sr

    @property
    def nav_df(self):
        if self.__nav_sr is None:
            return None
        else:
            nav_df = self.__nav_sr.to_frame()
            nav_df.columns = [self.__globalid]
        return nav_df

    @property
    def pos(self):
        return self.__pos

    @property
    def bound(self):
        return self.__bound


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



    def load_allocate_data(self, day ,asset_ids):

        bound_limit = self.bound.get_day_bound(day).loc[asset_ids]
        bound_limit = bound_limit[bound_limit.upper > 0.0]
        asset_ids_tmp = bound_limit.index
        reindex = self.index[self.index <= day][-1 * self.lookback:]
        data = {}
        for asset_id in asset_ids_tmp:
            data[asset_id] = self.assets[asset_id].nav(reindex = reindex)
        df_nav = pd.DataFrame(data).fillna(method='pad')
        df_inc  = df_nav.pct_change().fillna(0.0)

        bound = []
        for asset_id in df_inc.columns:
            bound.append(bound_limit.loc[asset_id].to_dict())

        return df_inc, bound


    def allocate_algo(self, day, df_inc, bound):

        ws = [1.0 / len(df_inc.columns)] * len(df_inc.columns)

        ws = dict(list(zip(df_inc.columns.ravel(), ws)))

        return ws


    def nav_update(self):
        return

    def turnover_update(self):
        return


    def load_db(self):
        pass

    def to_db(self):
        pass


if __name__ == '__main__':

    asset = Asset('120000001')
    print(asset.nav().tail())

    asset = WaveletAsset('120000013', 2)
    print(asset.nav('1900-01-01', datetime.now()).tail())

    trade_date = ATradeDate.week_trade_date(begin_date = '2012-01-01')
    print(trade_date[-5:])

    asset_globalids = ['120000001', '120000002', '120000013', '120000014', '120000015']
    assets = {}
    for asset_id in asset_globalids:
        assets[asset_id] = Asset(asset_id)

    allocate = Allocate('ALC.000001', assets, trade_date, 26)
    allocate.allocate()
