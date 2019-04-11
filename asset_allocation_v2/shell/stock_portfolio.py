#coding=utf-8
'''
Created on: Mar. 11, 2019
Modified on: Apr. 11, 2019
Author: Shixun Su, Boyang Zhou
Contact: sushixun@licaimofang.com
'''

import sys
import logging
import warnings
import click
from functools import partial
import multiprocessing
import numpy as np
import pandas as pd
import math
import hashlib
import types
import re
import copy
# from ipdb import set_trace
sys.path.append('shell')
from db import caihui_tq_ix_comp, caihui_tq_qt_index, caihui_tq_qt_skdailyprice, caihui_tq_sk_basicinfo, caihui_tq_sk_dquoteindic, caihui_tq_sk_finindic, caihui_tq_sk_sharestruchg
from db import factor_ml_merge_list, factor_sp_stock_portfolio_nav
from trade_date import ATradeDate
from util_timestamp import *


logger = logging.getLogger(__name__)


class MetaClassPropertyFuncGenerater(type):

    def __new__(cls, name, bases, attrs):

        attrs.update({variable: property(MetaClassPropertyFuncGenerater.generate_func_for_variable_in_data(variable)) \
            for variable in attrs.get('_variable_list_in_data', [])})
        attrs.update({variable: property(MetaClassPropertyFuncGenerater.generate_func_for_instance_variable(variable)) \
            for variable in attrs.get('_instance_variable_list', [])})

        return type.__new__(cls, name, bases, attrs)

    @staticmethod
    def generate_func_for_variable_in_data(variable):

        def func(self):

            if not hasattr(self, '_data'):
                raise AttributeError(f'\'{self.__class__.__name__}\'object has no attribute \'{variable}\'')
            else:
                data = getattr(self, '_data')
                if not hasattr(data, variable):
                    raise AttributeError(f'\'{self.__cls__.__name__}\'object has no attribute \'{variable}\'')

            return getattr(data, variable)

        return func

    @staticmethod
    def generate_func_for_instance_variable(variable):

        def func(self):

            if not hasattr(self, f'_{variable}'):
                raise AttributeError(f'\'{self.__cls__.__name__}\'object has no attribute \'{variable}\'')

            return getattr(self, f'_{variable}')

        return func


class StockPortfolioData:

    def __init__(self, index_id, reindex, look_back):

        self.index_id = index_id

        self.reindex = reindex
        self.look_back = look_back
        self.reindex_total = ATradeDate.trade_date(
            begin_date=self.reindex[0],
            end_date=self.reindex[-1],
            lookback=self.look_back+1
        ).rename('trade_date')

        self.df_index_historical_constituents = caihui_tq_ix_comp.load_index_historical_constituents(
            self.index_id,
            begin_date=self.reindex[0].strftime('%Y%m%d'),
            end_date=self.reindex[-1].strftime('%Y%m%d')
        )
        self.stock_pool = self.df_index_historical_constituents.loc[:, ['stock_id', 'stock_code']] \
            .drop_duplicates(subset=['stock_id']).set_index('stock_id').sort_index()

        df_merge_info = factor_ml_merge_list.load()
        self.df_merge_info = df_merge_info.loc[
            (df_merge_info.trade_date.isin(self.reindex)) & \
            (df_merge_info.old_stock_id.isin(self.stock_pool.index))
        ]

        stock_pool_total = pd.concat([self.stock_pool, caihui_tq_sk_basicinfo.load_stock_code_info(self.df_merge_info.new_stock_id)])
        self.stock_pool_total = stock_pool_total.loc[~stock_pool_total.index.duplicated()]

        self.df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_price(
            stock_ids=self.stock_pool_total.index,
            reindex=self.reindex_total
        )
        self.df_stock_ret = self.df_stock_prc.pct_change().iloc[1:]

        self.df_stock_status = caihui_tq_qt_skdailyprice.load_stock_status(
            stock_ids=self.stock_pool_total.index,
            reindex=self.reindex_total
        )

        self.df_stock_industry = caihui_tq_sk_basicinfo.load_stock_industry(stock_ids=self.stock_pool.index)
        self.df_stock_historical_share = self.__load_stock_historical_share()

        self.stock_market_data = caihui_tq_sk_dquoteindic.load_stock_market_data(
            stock_ids=self.stock_pool.index,
            reindex=self.reindex_total
        )

        self.stock_financial_data = caihui_tq_sk_finindic.load_stock_financial_data(
            stock_ids=self.stock_pool.index,
            reindex=self.reindex_total
        )

        self.ser_index_nav = caihui_tq_qt_index.load_index_nav(
            index_ids=[self.index_id],
            reindex=self.reindex_total
        )[self.index_id]

    def __load_stock_historical_share(self):

        ser_stock_company_id = caihui_tq_sk_basicinfo.load_stock_company_id(stock_ids=self.stock_pool.index) \
            .reset_index().set_index('company_id').stock_id

        df_stock_historical_share = caihui_tq_sk_sharestruchg.load_company_historical_share(
            company_ids=ser_stock_company_id.index,
            begin_date=self.reindex[0].strftime('%Y%m%d'),
            end_date=self.reindex[-1].strftime('%Y%m%d')
        )
        df_stock_historical_share['stock_id'] = df_stock_historical_share.company_id.map(ser_stock_company_id)
        df_stock_historical_share.drop('company_id', axis='columns', inplace=True)

        return df_stock_historical_share


class StockPortfolio(metaclass=MetaClassPropertyFuncGenerater):

    _ref_list = {}

    _variable_list_in_data = [
        'index_id',
        'reindex',
        'look_back',
        'reindex_total',
        'df_index_historical_constituents',
        'stock_pool',
        'df_merge_info',
        'stock_pool_total',
        'df_stock_prc',
        'df_stock_ret',
        'df_stock_status',
        'df_stock_industry',
        'df_stock_historical_share',
        'stock_market_data',
        'stock_financial_data'
    ]

    _instance_variable_list = [
        'df_stock_pos',
        'df_stock_pos_adjusted',
        'ser_portfolio_nav',
        'ser_portfolio_inc',
        'ser_turnover'
    ]

    _kwargs_list = []

    def __init__(self, index_id, reindex, look_back, **kwargs):

        ref = f'{index_id}, {reindex}, {look_back}'
        sha1 = hashlib.sha1()
        sha1.update(ref.encode('utf-8'))
        ref = sha1.hexdigest()

        if ref in StockPortfolio._ref_list:
            self._data = StockPortfolio._ref_list[ref]
        else:
            self._data = StockPortfolio._ref_list[ref] = StockPortfolioData(index_id, reindex, look_back)

        for key, value in kwargs.items():
            setattr(self, f'_{key}', value)
            setattr(self.__class__, key, property(MetaClassPropertyFuncGenerater.generate_func_for_instance_variable(key)))

        for key in self._kwargs_list:
            if not hasattr(self, key):
                raise TypeError(f'__init__() missing 1 required positional argument: \'{key}\'')

        self._df_stock_pos = None
        self._df_stock_pos_adjusted = None

        self._ser_portfolio_nav = None
        self._ser_portfolio_inc = None
        self._ser_turnover = None

    def calc_portfolio_nav(self, considering_status=True):

        self.calc_stock_pos_days()

        # df_stock_pos_adjusted = pd.DataFrame(index=self.reindex, columns=self.stock_pool.index)
        # df_stock_pos_adjusted.loc[self.reindex[0]] = self.df_stock_pos.loc[self.reindex[0], :].fillna(0.0)
        stock_pos_adjusted = self.df_stock_pos.loc[self.reindex[0], :].reindex(self.stock_pool_total.index).fillna(0.0)
        arr_stock_pos_adjusted = np.array(stock_pos_adjusted.values.reshape(1, -1))
        ser_portfolio_nav = pd.Series(1.0, index=self.reindex, name='nav')
        ser_portfolio_inc = pd.Series(0.0, index=self.reindex, name='inc')
        ser_turnover = pd.Series(0.0, index=self.reindex, name='turnover')

        for last_trade_date, trade_date in zip(self.reindex[:-1], self.reindex[1:]):

            stock_pos = stock_pos_adjusted * (self.df_stock_prc.loc[trade_date] / self.df_stock_prc.loc[last_trade_date]).fillna(1.0)

            nav = stock_pos.sum()
            ser_portfolio_inc.loc[trade_date] = nav - 1.0
            ser_portfolio_nav.loc[trade_date] = ser_portfolio_nav.loc[last_trade_date] * nav

            stock_pos.loc[:] = stock_pos / nav
            stock_pos_standard = self.df_stock_pos.loc[trade_date].fillna(0.0)
            if considering_status:
                stock_status = self.df_stock_status.loc[trade_date]
            else:
                stock_status = pd.Series(0, index=self.stock_pool_total.index, name=trade_date)

            index_adjustable_stock = stock_status.loc[stock_status==0].index
            sum_pos_adjustable = 1.0 - stock_pos.loc[stock_status>0].sum()
            sum_pos_standard = 1.0 - stock_pos_standard.loc[stock_status>0].sum()

            stock_pos_adjusted = copy.deepcopy(stock_pos)
            stock_pos_adjusted.loc[index_adjustable_stock] = stock_pos_standard.loc[index_adjustable_stock] * sum_pos_adjustable / sum_pos_standard
            ser_turnover.loc[trade_date] = (stock_pos_adjusted - stock_pos).abs().sum()

            if considering_status:
                for _, merge_info in self.df_merge_info.loc[self.df_merge_info.trade_date==trade_date].iterrows():

                    pos = stock_pos_adjusted.loc[merge_info.old_stock_id] * \
                        (merge_info.ratio * merge_info.new_stock_price / merge_info.old_stock_price)

                    stock_pos_adjusted.loc[merge_info.new_stock_id] = pos
                    stock_pos_adjusted.loc[merge_info.old_stock_id] = 0.0

            # df_stock_pos_adjusted.loc[trade_date] = stock_pos_adjusted
            arr_stock_pos_adjusted = np.append(arr_stock_pos_adjusted, stock_pos_adjusted.values.reshape(1, -1), axis=0)

        df_stock_pos_adjusted = pd.DataFrame(arr_stock_pos_adjusted, index=self.reindex, columns=self.stock_pool_total.index)

        ser0_portfolio_inc = pd.Series(0.0, index=self.reindex, name='inc')
        ser0_portfolio_nav = pd.Series(1.0, index=self.reindex, name='nav')
        ser0_portfolio_inc.loc[:] = (df_stock_pos_adjusted.rename(index=trade_date_after) * (self.df_stock_prc / self.df_stock_prc.rename(index=trade_date_after) - 1.0)).sum(axis='columns')
        ser0_portfolio_nav.loc[:] = (ser0_portfolio_inc + 1.0).cumprod()

        self._df_stock_pos_adjusted = df_stock_pos_adjusted
        self._ser_portfolio_nav = ser_portfolio_nav
        self._ser_portfolio_inc = ser_portfolio_inc
        self._ser_turnover = ser_turnover

        return ser_portfolio_nav

    def calc_stock_pos_days(self):

        if self.df_stock_pos is not None:
            return self.df_stock_pos

        ser_reindex = pd.Series(self.reindex, index=self.reindex)
        df_stock_pos = pd.DataFrame(index=self.reindex, columns=self.stock_pool.index)
        df_stock_pos.loc[:, :] = ser_reindex.apply(self._calc_stock_pos)

        self._df_stock_pos = df_stock_pos

        return df_stock_pos

    def _calc_stock_pos(self, trade_date):

        raise NotImplementedError('Method \'calc_stock_pos\' is not defined.')

    def _load_stock_price(self, trade_date, stock_ids):

        reindex = self.reindex_total[self.reindex_total<=trade_date][-self.look_back-1:]

        # df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_prc(stock_ids=stock_ids, reindex=reindex)
        df_stock_prc = self.df_stock_prc.reindex(index=reindex, columns=stock_ids)

        return df_stock_prc

    def _load_stock_return(self, trade_date, stock_ids):

        reindex = self.reindex_total[self.reindex_total<=trade_date][-self.look_back:]

        # df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_prc(stock_ids=stock_ids, reindex=reindex)
        # df_stock_ret = df_stock_prc.pct_change().iloc[1:]
        df_stock_ret = self.df_stock_ret.reindex(index=reindex, columns=stock_ids)

        return df_stock_ret

    def _load_stock_pool(self, trade_date):

        trade_date = trade_date_after(trade_date)

        # stock_pool = caihui_tq_ix_comp.load_index_constituents(self.index_id, date=trade_date.strftime('%Y%m%d'))
        stock_pool = self.df_index_historical_constituents.loc[
            (self.df_index_historical_constituents.selected_date<=trade_date) & \
            ((self.df_index_historical_constituents.out_date>trade_date) | \
            (self.df_index_historical_constituents.out_date=='19000101'))
        ].loc[:, ['stock_id', 'stock_code']].set_index('stock_id').sort_index()

        return stock_pool

    def portfolio_analysis(self, reindex=None):

        if self.ser_portfolio_nav is None:
            self.calc_portfolio_nav()

        if reindex is None:
            reindex = self.reindex

        portfolio_return = self.ser_portfolio_nav.loc[reindex[-1]] / self.ser_portfolio_nav.loc[reindex[0]] - 1.0
        free_risk_rate = 0.0
        std_excess_return = self.ser_portfolio_inc.reindex(reindex).std()
        sharpe_ratio = (portfolio_return - free_risk_rate) / std_excess_return

        return portfolio_return, std_excess_return, sharpe_ratio


class StockPortfolioMarketCap(StockPortfolio):

    _kwargs_list = []

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioMarketCap, self).__init__(index_id, reindex, look_back, **kwargs)

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/000300
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        _, ser_market_cap_stock_weight = self._calc_market_cap(trade_date, stock_ids)

        stock_pos = pd.Series(index=stock_ids, name=trade_date)
        stock_pos.loc[:] = (ser_market_cap_stock_weight).fillna(0.0)

        return stock_pos

    def _calc_market_cap(self, trade_date, stock_ids, **kwargs):

        ser_stock_free_float_market_cap = self._calc_stock_free_float_market_cap(trade_date, stock_ids)

        ser_market_cap_stock_weight = ser_stock_free_float_market_cap
        ser_market_cap_stock_weight /= ser_market_cap_stock_weight.sum()

        return stock_ids, ser_market_cap_stock_weight

    def _calc_stock_free_float_market_cap(self, trade_date, stock_ids):

        df_stock_share = self._load_stock_share(trade_date, stock_ids)
        ser_stock_total_market_cap = self.stock_financial_data['total_market_cap'].loc[trade_date, stock_ids]

        ser_free_float_weight = df_stock_share.free_float_share / df_stock_share.total_share
        ser_free_float_weight.loc[:] = ser_free_float_weight.apply(self._weight_adjustment_algo)

        ser_stock_free_float_market_cap = (ser_stock_total_market_cap * ser_free_float_weight).rename('free_float_market_cap')

        return ser_stock_free_float_market_cap

    def _load_stock_share(self, trade_date, stock_ids):

        df_stock_share = self.df_stock_historical_share.loc[
            (self.df_stock_historical_share.begin_date<=trade_date) & \
            ((self.df_stock_historical_share.end_date>=trade_date) | \
            (self.df_stock_historical_share.end_date=='19000101'))
        ].sort_values(by='begin_date').drop_duplicates(subset=['stock_id'], keep='last').set_index('stock_id').reindex(stock_ids)

        return df_stock_share

    def _weight_adjustment_algo(self, weight):

        if weight > 0.8:
            return 1.0
        elif weight > 0.15:
            return math.ceil(weight * 10.0) / 10.0
        elif weight > 0.0:
            return math.ceil(weight * 100.0) / 100.0
        else:
            return np.nan


class StockPortfolioEqualWeight(StockPortfolio):

    _kwargs_list = []

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioEqualWeight, self).__init__(index_id, reindex, look_back, **kwargs)

    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        stock_pos = pd.Series(1.0/stock_ids.size, index=stock_ids, name=trade_date)

        return stock_pos


class StockPortfolioLowVolatility(StockPortfolio):

    _kwargs_list = [
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioLowVolatility, self).__init__(index_id, reindex, look_back, **kwargs)

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/000803
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        low_volatility_stock_ids, ser_low_volatility_stock_weight = self._calc_low_volatility(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[low_volatility_stock_ids] = ser_low_volatility_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_low_volatility(self, trade_date, stock_ids, percentage, **kwargs):

        df_stock_ret = self._load_stock_return(trade_date, stock_ids)
        df_stock_status = self.df_stock_status.loc[df_stock_ret.index, stock_ids]
        df_stock_ret[df_stock_status>2] = np.nan

        ser_stock_volatility = df_stock_ret.std()

        # Low volatility condition quantile as percentage
        portfolio_size = round(stock_ids.size * percentage)
        low_volatility_stock_ids = ser_stock_volatility.sort_values(ascending=True).iloc[:portfolio_size].index

        ser_low_volatility_stock_weight = 1.0 / ser_stock_volatility.loc[low_volatility_stock_ids]
        ser_low_volatility_stock_weight /= ser_low_volatility_stock_weight.sum()

        return low_volatility_stock_ids, ser_low_volatility_stock_weight


class StockPortfolioMomentum(StockPortfolioMarketCap):

    _kwargs_list = [
        'percentage',
        'exclusion'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioMomentum, self).__init__(index_id, reindex, look_back, **kwargs)

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/H30260
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        momentum_stock_ids, _ = self._calc_momentum(trade_date, stock_ids, self.percentage)
        _, ser_market_cap_stock_weight = self._calc_market_cap(trade_date, momentum_stock_ids)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[momentum_stock_ids] = ser_market_cap_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_momentum(self, trade_date, stock_ids, percentage, **kwargs):

        df_stock_prc = self._load_stock_price(trade_date, stock_ids)

        ser_stock_momentum = df_stock_prc.iloc[-1-self.exclusion] / df_stock_prc.iloc[0]

        # Momentum condition quantile as percentage
        portfolio_size = round(stock_ids.size * percentage)
        momentum_stock_ids = ser_stock_momentum.sort_values(ascending=False).iloc[:portfolio_size].index

        ser_momentum_stock_weight = ser_stock_momentum.loc[momentum_stock_ids]
        ser_momentum_stock_weight /= ser_momentum_stock_weight.sum()

        return momentum_stock_ids, ser_momentum_stock_weight


class StockPortfolioSmallSize(StockPortfolioMarketCap):

    _kwargs_list = [
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioSmallSize, self).__init__(index_id, reindex, look_back, **kwargs)

    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        small_size_stock_ids, ser_small_size_stock_weight = self._calc_small_size(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[small_size_stock_ids] = ser_small_size_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_small_size(self, trade_date, stock_ids, percentage, **kwargs):

        ser_stock_free_float_market_cap = self._calc_stock_free_float_market_cap(trade_date, stock_ids)

        portfolio_size = round(stock_ids.size * self.percentage)
        small_size_stock_ids = ser_stock_free_float_market_cap.sort_values(ascending=True).iloc[:portfolio_size].index

        ser_small_size_stock_weight = ser_stock_free_float_market_cap.loc[small_size_stock_ids]
        ser_small_size_stock_weight /= ser_small_size_stock_weight.sum()

        return small_size_stock_ids, ser_small_size_stock_weight


class StockPortfolioLowBeta(StockPortfolio):

    _kwargs_list = [
        'percentage',
        'benchmark_id',
    ]

    _instance_variable_list = StockPortfolio._instance_variable_list + ['ser_benchmark_nav', 'ser_benchmark_inc']

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioLowBeta, self).__init__(index_id, reindex, look_back, **kwargs)

        df_benchmark_nav = factor_sp_stock_portfolio_nav.load(self.benchmark_id)
        self._ser_benchmark_nav = df_benchmark_nav.nav.rename(self.benchmark_id)
        self._ser_benchmark_inc = df_benchmark_nav.inc.rename(self.benchmark_id)

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/000829
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        low_beta_stock_ids, ser_low_beta_stock_weight = self._calc_low_beta(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[low_beta_stock_ids] = ser_low_beta_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_low_beta(self, trade_date, stock_ids, percentage, **kwargs):

        df_stock_ret = self._load_stock_return(trade_date, stock_ids)
        df_stock_status = self.df_stock_status.loc[df_stock_ret.index, stock_ids]
        df_stock_ret[df_stock_status>2] = np.nan

        ser_benchmark_inc = self.ser_benchmark_inc.loc[df_stock_ret.index]
        benchmark_inc_std = ser_benchmark_inc.std()

        calc_stock_benchmark_cov = partial(self._calc_series_cov, ser_benchmark_inc)
        ser_stock_beta = df_stock_ret.apply(calc_stock_benchmark_cov) / (benchmark_inc_std ** 2)

        # Low beta condition quantile as percentage
        portfolio_size = round(stock_ids.size * percentage)
        low_beta_stock_ids = ser_stock_beta.sort_values(ascending=True).iloc[:portfolio_size].index

        ser_low_beta_stock_weight = 1.0 / ser_stock_beta.loc[low_beta_stock_ids]
        ser_low_beta_stock_weight /= ser_low_beta_stock_weight.sum()

        return low_beta_stock_ids, ser_low_beta_stock_weight

    def _calc_series_cov(self, ser1, ser2):

        df = pd.DataFrame({ser1.name: ser1, ser2.name: ser2})
        ser_cov = df.cov().iloc[0, 1]

        return ser_cov


class StockPortfolioHighBeta(StockPortfolioLowBeta):

    _kwargs_list = [
        'percentage',
        'benchmark_id'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioHighBeta, self).__init__(index_id, reindex, look_back, **kwargs)

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/000828
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        high_beta_stock_ids, ser_high_beta_stock_weight = self._calc_high_beta(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[high_beta_stock_ids] = ser_high_beta_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_high_beta(self, trade_date, stock_ids, percentage, **kwargs):

        df_stock_ret = self._load_stock_return(trade_date, stock_ids)
        df_stock_status = self.df_stock_status.loc[df_stock_ret.index, stock_ids]
        df_stock_ret[df_stock_status>2] = np.nan

        ser_benchmark_inc = self.ser_benchmark_inc.loc[df_stock_ret.index]
        benchmark_inc_std = ser_benchmark_inc.std()

        calc_stock_benchmark_cov = partial(self._calc_series_cov, ser_benchmark_inc)
        ser_stock_beta = df_stock_ret.apply(calc_stock_benchmark_cov) / (benchmark_inc_std ** 2)

        # High beta condition quantile as percentage
        portfolio_size = round(stock_ids.size * percentage)
        high_beta_stock_ids = ser_stock_beta.sort_values(ascending=False).iloc[:portfolio_size].index

        ser_high_beta_stock_weight = ser_stock_beta.loc[high_beta_stock_ids]
        ser_high_beta_stock_weight /= ser_high_beta_stock_weight.sum()

        return high_beta_stock_ids, ser_high_beta_stock_weight


class StockPortfolioHighBetaAndLowBeta(StockPortfolioHighBeta):

    _kwargs_list = [
        'percentage',
        'benchmark_id',
        'factor_weight'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioHighBetaAndLowBeta, self).__init__(index_id, reindex, look_back, **kwargs)

    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        high_beta_and_low_beta_stock_ids, ser_high_beta_and_low_beta_stock_weight = self._calc_high_beta_and_low_beta(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[high_beta_and_low_beta_stock_ids] = ser_high_beta_and_low_beta_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_high_beta_and_low_beta(self, trade_date, stock_ids, percentage, **kwargs):

        high_beta_stock_ids, ser_high_beta_stock_weight = self._calc_high_beta(trade_date, stock_ids, self.percentage / 2.0)
        low_beta_stock_ids, ser_low_beta_stock_weight = self._calc_low_beta(trade_date, stock_ids, self.percentage / 2.0)

        high_beta_and_low_beta_stock_ids = high_beta_stock_ids.union(low_beta_stock_ids)

        ser_high_beta_and_low_beta_stock_weight = pd.Series(0.0, index=high_beta_and_low_beta_stock_ids, name=trade_date)
        ser_high_beta_and_low_beta_stock_weight.loc[high_beta_stock_ids] += ser_high_beta_stock_weight * self.factor_weight['high_beta']
        ser_high_beta_and_low_beta_stock_weight.loc[low_beta_stock_ids] += ser_low_beta_stock_weight * self.factor_weight['low_beta']
        ser_high_beta_and_low_beta_stock_weight /= ser_high_beta_and_low_beta_stock_weight.sum()

        return high_beta_and_low_beta_stock_ids, ser_high_beta_and_low_beta_stock_weight


class StockPortfolioLowBetaLowVolatility(StockPortfolioLowBeta, StockPortfolioLowVolatility):

    _kwargs_list = [
        'percentage_low_beta',
        'percentage_low_volatility',
        'benchmark_id'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioLowBetaLowVolatility, self).__init__(index_id, reindex, look_back, **kwargs)

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/930985
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        low_beta_stock_ids, _ = self._calc_low_beta(trade_date, stock_ids, self.percentage_low_beta)
        low_beta_low_volatility_stock_ids, _ = self._calc_low_volatility(trade_date, low_beta_stock_ids, self.percentage_low_volatility)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[low_beta_low_volatility_stock_ids] = 1.0 / low_beta_low_volatility_stock_ids.size

        return stock_pos


class StockPortfolioSectorNeutral(StockPortfolioMarketCap):

    _kwargs_list = []

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioSectorNeutral, self).__init__(index_id, reindex, look_back, **kwargs)

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/930846
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        _, ser_sector_neutral_stock_weight = self._calc_sector_neutral(trade_date, stock_ids)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[stock_ids] = ser_sector_neutral_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_sector_neutral(self, trade_date, stock_ids, **kwargs):

        ser_industry_free_float_market_cap = self._calc_industry_free_float_market_cap(trade_date, stock_ids)

        ser_sector_neutral_stock_weight = pd.Series(index=stock_ids, name=trade_date)

        for sw_industry_code in ser_industry_free_float_market_cap.index:

            sw_industry_stock_ids = stock_ids.intersection(self.df_stock_industry.index[self.df_stock_industry.sw_level1_code==sw_industry_code])

            class_name = self.__class__.__name__
            algo = re.sub('StockPortfolioSectorNeutral', '', class_name, count=1)
            func_name = '_calc' + re.sub('(?P<l>[A-Z]+)', lambda x: '_'+x.group('l').lower(), algo)

            if not hasattr(self, func_name):
                raise AttributeError(f'\'{class_name}\'object has no attribute \'{func_name}\'')
            calc_func = getattr(self, func_name)
            _, ser_stock_weight_by_industry = calc_func(trade_date, sw_industry_stock_ids, percentage=1.0)

            ser_sector_neutral_stock_weight.loc[sw_industry_stock_ids] = ser_stock_weight_by_industry * ser_industry_free_float_market_cap.loc[sw_industry_code]

        ser_sector_neutral_stock_weight.loc[:] /= ser_sector_neutral_stock_weight.sum()

        return stock_ids, ser_sector_neutral_stock_weight

    def _calc_industry_free_float_market_cap(self, trade_date, stock_ids, **kwargs):

        ser_stock_free_float_market_cap = self._calc_stock_free_float_market_cap(trade_date, stock_ids)
        df_stock_industry = self.df_stock_industry.loc[stock_ids]

        ser_industry_free_float_market_cap = ser_stock_free_float_market_cap.rename(index=df_stock_industry.sw_level1_code).rename_axis('sw_industry_code').groupby(by='sw_industry_code').sum()

        return ser_industry_free_float_market_cap


class StockPortfolioSectorNeutralLowVolatility(StockPortfolioSectorNeutral, StockPortfolioLowVolatility):

    _kwargs_list = []

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioSectorNeutralLowVolatility, self).__init__(index_id, reindex, look_back, **kwargs)


class StockPortfolioSectorNeutralLowBeta(StockPortfolioSectorNeutral, StockPortfolioLowBeta):

    _kwargs_list = [
        'benchmark_id'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioSectorNeutralLowBeta, self).__init__(index_id, reindex, look_back, **kwargs)


class StockPortfolioIndustry(StockPortfolio):

    _kwargs_list = [
        'sw_industry_code'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioIndustry, self).__init__(index_id, reindex, look_back, **kwargs)

    def _calc_stock_pos(self, trade_date):

        sw_industry_stock_ids = self._load_stock_pool(trade_date).index

        class_name = self.__class__.__name__
        algo = re.sub('StockPortfolioIndustry', '', class_name, count=1)
        func_name = '_calc' + re.sub('(?P<l>[A-Z]+)', lambda x: '_'+x.group('l').lower(), algo)

        if not hasattr(self, func_name):
            raise AttributeError(f'\'{class_name}\'object has no attribute \'{func_name}\'')
        calc_func = getattr(self, func_name)
        _, ser_stock_weight_by_industry = calc_func(trade_date, sw_industry_stock_ids, percentage=1.0)

        stock_pos = pd.Series(0.0, index=sw_industry_stock_ids, name=trade_date)
        stock_pos.loc[sw_industry_stock_ids] = ser_stock_weight_by_industry

        return stock_pos

    def _load_stock_pool(self, trade_date):

        stock_pool = super(StockPortfolioIndustry, self)._load_stock_pool(trade_date)

        stock_ids_by_industry = self._load_stock_ids_by_industry()
        stock_pool = stock_pool.reindex(stock_ids_by_industry)

        return stock_pool

    def _load_stock_ids_by_industry(self):

        stock_ids_by_industry = self.df_stock_industry.index[self.df_stock_industry.sw_level1_code==self.sw_industry_code]

        return stock_ids_by_industry


class StockPortfolioIndustryLowVolatility(StockPortfolioIndustry, StockPortfolioLowVolatility):

    _kwargs_list = [
        'sw_industry_code'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioIndustryLowVolatility, self).__init__(index_id, reindex, look_back, **kwargs)


class StockPortfolioIndustryMomentum(StockPortfolioIndustry, StockPortfolioMomentum):

    _kwargs_list = [
        'sw_industry_code',
        'exclusion'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioIndustryMomentum, self).__init__(index_id, reindex, look_back, **kwargs)


def func(algo, index_id, trade_dates, look_back, sw_industry_code, **kwargs):

    class_name = f'StockPortfolio{algo}'
    cls = globals()[class_name]

    kwargs['sw_industry_code'] = sw_industry_code

    df = pd.DataFrame()

    portfolio = cls(index_id, trade_dates, look_back, **kwargs)
    df['sw_industry_code'] = portfolio.calc_portfolio_nav()

    df.to_csv(f'{algo}_{sw_industry_code}.csv')

def multiprocessing_calc_portfolio_nav_by_industry(algo, index_id, trade_dates, look_back, *args, **kwargs):

    sw_industry_codes = caihui_tq_sk_basicinfo.load_sw_industry_code_info().index

    portfolio = StockPortfolio(index_id, trade_dates, look_back)

    for sw_industry_code in sw_industry_codes:

        process = multiprocessing.Process(
            target=func,
            args=(algo, index_id, trade_dates, look_back, sw_industry_code),
            kwargs={**kwargs}
        )
        process.start()


if __name__ == '__main__':

    index_id = '2070000191'
    begin_date = '2018-12-28'
    end_date = '2019-03-29'
    look_back = 244

    trade_dates = ATradeDate.trade_date(begin_date=begin_date, end_date=end_date).rename('trade_date')

    dict_portfolio = {}
    df_portfolio_nav = pd.DataFrame()

    # dict_portfolio['MarketCap'] = StockPortfolioMarketCap(index_id, trade_dates, look_back)
    # df_portfolio_nav['MarketCap'] = dict_portfolio['MarketCap'].calc_portfolio_nav()

    # dict_portfolio['EqualWeight'] = StockPortfolioEqualWeight(index_id, trade_dates, look_back)
    # df_portfolio_nav['EqualWeight'] = dict_portfolio['EqualWeight'].calc_portfolio_nav()

    # dict_portfolio['LowVolatility'] = StockPortfolioLowVolatility(index_id, trade_dates, look_back, percentage=0.3)
    # df_portfolio_nav['LowVolatility'] = dict_portfolio['LowVolatility'].calc_portfolio_nav()

    # dict_portfolio['Momentum'] = StockPortfolioMomentum(index_id, trade_dates, look_back, percentage=0.3, exclusion=20)
    # df_portfolio_nav['Momentum'] = dict_portfolio['Momentum'].calc_portfolio_nav()

    # dict_portfolio['SmallSize'] = StockPortfolioSmallSize(index_id, trade_dates, look_back, percentage=0.3)
    # df_portfolio_nav['SmallSize'] = dict_portfolio['SmallSize'].calc_portfolio_nav()

    # dict_portfolio['LowBeta'] = StockPortfolioLowBeta(index_id, trade_dates, look_back, percentage=0.3, benchmark_id='CS.000906')
    # df_portfolio_nav['LowBeta'] = dict_portfolio['LowBeta'].calc_portfolio_nav()

    # dict_portfolio['HighBeta'] = StockPortfolioHighBeta(index_id, trade_dates, look_back, percentage=0.3, benchmark_id='CS.000906')
    # df_portfolio_nav['HighBeta'] = dict_portfolio['HighBeta'].calc_portfolio_nav()

    # dict_portfolio['HighBetaAndLowBeta'] = StockPortfolioHighBetaAndLowBeta(index_id, trade_dates, look_back, percentage=0.6, benchmark_id='CS.000906', factor_weight={'high_beta': 0.5, 'low_beta':0.5})
    # df_portfolio_nav['HighBetaAndLowBeta'] = dict_portfolio['HighBetaAndLowBeta'].calc_portfolio_nav()

    # dict_portfolio['LowBetaLowVolatility'] = StockPortfolioLowBetaLowVolatility(index_id, trade_dates, look_back, percentage_low_beta=0.6, percentage_low_volatility=0.5, benchmark_id='CS.000906')
    # df_portfolio_nav['LowBetaLowVolatility'] = dict_portfolio['LowBetaLowVolatility'].calc_portfolio_nav()

    # dict_portfolio['SectorNeutralLowVolatility'] = StockPortfolioSectorNeutralLowVolatility(index_id, trade_dates, look_back)
    # df_portfolio_nav['SectorNeutralLowVolatility'] = dict_portfolio['SectorNeutralLowVolatility'].calc_portfolio_nav()

    # dict_portfolio['SectorNeutralLowBeta'] = StockPortfolioSectorNeutralLowBeta(index_id, trade_dates, look_back, benchmark_id='CS.000905')
    # df_portfolio_nav['SectorNeutralLowBeta'] = dict_portfolio['SectorNeutralLowBeta'].calc_portfolio_nav()

    # df_portfolio_nav.to_csv('df_portfolio_nav.csv')
    # set_trace()

    # multiprocessing_calc_portfolio_nav_by_industry('IndustryLowVolatility', index_id, trade_dates, look_back, percentage=0.30)

    # multiprocessing_calc_portfolio_nav_by_industry('IndustryMomentum', index_id, trade_dates, look_back, percentage=0.3, exclusion=30)
    # set_trace()

