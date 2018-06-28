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
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates, base_exchange_rate_index_nav, asset_ra_bl
from util import xdict
from util.xdebug import dd
from wavelet import Wavelet
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
import db.asset_stock

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
    def name(self):
        return self.__name

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
            xtype = int(asset_id) // 10000000
            if xtype == 1:
                #
                # 基金池资产
                #
                asset_id = int(asset_id) % 10000000
                (pool_id, category) = (asset_id // 100, asset_id % 100)
                ttype = pool_id // 10000
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
            elif prefix == 'SK':
                #
                # 股票资产
                #
                sr = db.asset_stock.load_stock_nav_series(
                    asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
            else:
                sr = pd.Series()

        return sr



class WaveletAsset(Asset):


    def __init__(self, globalid, wavelet_filter_num, name = None, nav_sr = None):

        super(WaveletAsset, self).__init__(globalid, name = name, nav_sr = nav_sr)
        self.__wavelet_filter_num = wavelet_filter_num


    @property
    def wavelet_filter_num(self):
        return self.__wavelet_filter_num


    def nav(self, wave_begin_date = None, wave_end_date = None, begin_date = None, end_date = None, reindex = None):

        if wave_begin_date is None:
            wave_begin_date = '1900-01-01'
        if wave_end_date is None:
            if end_date is not None:
                wave_end_date = end_date
            elif reindex is not None:
                reindex = reindex.sort_values()
                wave_end_date = reindex[-1]

        nav_sr = super(WaveletAsset, self).nav(begin_date = wave_begin_date, end_date = wave_end_date)

        wavelet_nav_sr = Wavelet.wavefilter(nav_sr, self.wavelet_filter_num)


        if begin_date is not None:
            wavelet_nav_sr = wavelet_nav_sr[wavelet_nav_sr.index >= begin_date]
        if reindex is not None:
            wavelet_nav_sr = wavelet_nav_sr.reindex(reindex).fillna(method = 'pad')

            wavelet_nav_sr = wavelet_nav_sr.loc[reindex]

        return wavelet_nav_sr


class StockAsset(Asset):

    __all_stock_info = None
    __all_st_stocks = None
    __all_stock_quote = {}
    __all_stock_fdmt = {}
    __all_stocks = {}

    def __init__(self, globalid, name = None, nav_sr = None):

        super(StockAsset, self).__init__(globalid, name = name, nav_sr = nav_sr)
        self.__secode = None
        self.__quote = None
        self.__fdmt = None
        self.__code = globalid[3:]


    @property
    def code(self):
        return self.__code


    @staticmethod
    def get_stock(globalid):
        if not globalid in StockAsset.__all_stocks:
            StockAsset.__all_stocks[globalid] = StockAsset(globalid)
        return StockAsset.__all_stocks[globalid]


    @staticmethod
    def all_stocks():
        stock_ids = list(set(StockAsset.all_stock_info().index.ravel()).difference(StockAsset.__all_stocks.keys()))
        if len(stock_ids) > 0:
            count = multiprocessing.cpu_count()
            pool = Pool(count // 2)
            results = pool.map(db.asset_stock.load_stock_nav_series, stock_ids)
            pool.close()
            pool.join()
            for i in range(0, len(stock_ids)):
                asset = StockAsset(stock_ids[i], nav_sr = results[i])
                StockAsset.__all_stocks[stock_ids[i]] = asset
        return StockAsset.__all_stocks


    @staticmethod
    def all_stock_nav():
        StockAsset.all_stocks()
        stock_nav = {}
        for stock_id in StockAsset.all_stock_info().index:
            asset = StockAsset.get_stock(stock_id)
            stock_nav[stock_id] = asset.nav().replace(0.0, method = 'pad')
        stock_nav_df = pd.DataFrame(stock_nav)
        return stock_nav_df


    @staticmethod
    def all_stock_amount():
        StockAsset.all_stocks()
        stock_amount = {}
        for stock_id in StockAsset.all_stock_info().index:
            asset = StockAsset.get_stock(stock_id)
            stock_amount[stock_id] = asset.quote.amount
        stock_amount_df = pd.DataFrame(stock_amount)
        return stock_amount_df


    @staticmethod
    def secode_dict():
        all_stock_info = StockAsset.all_stock_info()
        return dict(zip(all_stock_info.index, all_stock_info.sk_secode))

    @staticmethod
    def compcode_dict():
        all_stock_info = StockAsset.all_stock_info()
        return dict(zip(all_stock_info.index, all_stock_info.sk_compcode))

    #所有股票行情
    @staticmethod
    def all_stock_quote():
        stock_ids = list(set(StockAsset.all_stock_info().index.ravel()).difference(StockAsset.__all_stock_quote.keys()))
        if len(stock_ids) > 0:
            count = multiprocessing.cpu_count()
            pool = Pool(count // 2)
            results = pool.map(db.asset_stock.load_ohlcavntt, stock_ids)
            pool.close()
            pool.join()
            StockAsset.__all_stock_quote.update(dict(zip(stock_ids, results)))
        return StockAsset.__all_stock_quote


    #所有股票基本面
    @staticmethod
    def all_stock_fdmt():
        stock_ids = list(set(StockAsset.all_stock_info().index.ravel()).difference(StockAsset.__all_stock_fdmt.keys()))
        if len(stock_ids) > 0:
            count = multiprocessing.cpu_count()
            pool = Pool(count // 2)
            results = pool.map(db.asset_stock.load_fdmt, stock_ids)
            pool.close()
            pool.join()
            StockAsset.__all_stock_fdmt.update(dict(zip(stock_ids, results)))
        return StockAsset.__all_stock_fdmt


    @property
    def quote(self):
        if not self.globalid in StockAsset.__all_stock_quote:
            quote_df = db.asset_stock.load_ohlcavntt(self.globalid)
            StockAsset.__all_stock_quote[self.globalid] = quote_df
        return StockAsset.__all_stock_quote[self.globalid]


    @property
    def fdmt(self):
        if not self.globalid in StockAsset.__all_stock_fdmt:
            fdmt_df = db.asset_stock.load_fdmt(self.globalid)
            StockAsset.__all_stock_fdmt[self.globalid] = fdmt_df
        return StockAsset.__all_stock_fdmt[self.globalid]



    #所有股票代码
    @staticmethod
    def all_stock_info(globalids = None):
        if StockAsset.__all_stock_info is None:
            engine = database.connection('base')
            Session = sessionmaker(bind=engine)
            session = Session()
            all_stocks = pd.read_sql(session.query(db.asset_stock.ra_stock.globalid, db.asset_stock.ra_stock.sk_secode, db.asset_stock.ra_stock.sk_compcode, db.asset_stock.ra_stock.sk_name, db.asset_stock.ra_stock.sk_listdate, db.asset_stock.ra_stock.sk_swlevel1code).statement, session.bind, index_col = ['globalid'])
            session.commit()
            session.close()
            StockAsset.__all_stock_info = all_stocks;
        if globalids is None:
            return StockAsset.__all_stock_info
        else:
            return StockAsset.__all_stock_info.loc[globalids]


    #st股票表
    @staticmethod
    def stock_st():
        if StockAsset.__all_st_stocks is None:
            all_stocks = StockAsset.all_stock_info()
            all_stocks = all_stocks.reset_index()
            all_stocks = all_stocks.set_index(['sk_secode'])

            engine = database.connection('caihui')
            Session = sessionmaker(bind=engine)
            session = Session()
            sql = session.query(db.asset_stock.tq_sk_specialtrade.secode, db.asset_stock.tq_sk_specialtrade.selecteddate,
                    db.asset_stock.tq_sk_specialtrade.outdate).filter(db.asset_stock.tq_sk_specialtrade.selectedtype <= 3).filter(db.asset_stock.tq_sk_specialtrade.secode.in_(set(all_stocks.index))).statement
            st_stocks = pd.read_sql(sql, session.bind, index_col = ['secode'], parse_dates = ['selecteddate', 'outdate'])
            session.commit()
            session.close()

            StockAsset.__all_st_stocks = pd.merge(st_stocks, all_stocks, left_index=True, right_index=True)

        return StockAsset.__all_st_stocks



if __name__ == '__main__':

    # asset = Asset('120000001')
    #print asset.nav(begin_date = '2010-01-01').head()
    #print asset.origin_nav_sr.head()

    # asset = WaveletAsset('120000013', 2)
    #print asset.nav('2010-01-01', datetime.now()).tail()
    #print asset.origin_nav_sr.tail()

    # asset = StockAsset('SK.601318')
    # print asset.nav()
    # print asset.name
    #StockAsset.all_stock_fdmt()
    #for globalid in StockAsset.all_stock_info().index:
    #    asset = StockAsset('SK.601318')
    #    print time.time()
    #    asset.fdmt
    #    print time.time()
    #    print
    #print asset.nav()
    #print asset.name
    #print asset.load_ohlcavntt()

    #print StockAsset.all_stock_info()
    # print StockAsset.stock_st()
    # set_trace()
    # print asset.load_quote().head()
    #print asset.load_fdmt().head()

    # print asset.load_quote().head()
    # print asset.load_fdmt().head()
    # print asset.load_fdmt().head()

    print(StockAsset.all_stock_quote().keys())
    print(len(StockAsset.all_stock_quote().keys()))
