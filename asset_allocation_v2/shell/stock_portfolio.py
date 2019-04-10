#coding=utf-8
'''
Created on: Mar. 11, 2019
Modified on: Apr. 8, 2019
Author: Shixun Su, Boyang Zhou
Contact: sushixun@licaimofang.com
'''

import sys
import logging
import warnings
import click
from sqlalchemy import MetaData, Table, select, func, and_
from functools import partial
import multiprocessing
import numpy as np
import pandas as pd
import math
import hashlib
import copy
# from ipdb import set_trace
sys.path.append('shell')
from db import caihui_tq_ix_comp, caihui_tq_qt_index, caihui_tq_qt_skdailyprice, caihui_tq_sk_basicinfo, caihui_tq_sk_dquoteindic, caihui_tq_sk_finindic, caihui_tq_sk_sharestruchg
from db import factor_ml_merge_list, factor_sp_stock_portfolio_nav
from trade_date import ATradeDate
from util_timestamp import *


logger = logging.getLogger(__name__)


# class MetaClassPropertyDecorator(type):

    # def __new__(cls, name, bases, attrs):

        # attrs.update({name: property(MetaClassPropertyDecorator.generate_property_func_in_data(name)) for name in attrs.get('_name_list_in_data', [])})
        # attrs.update({name: property(MetaClassPropertyDecorator.generate_property_func_private(name)) for name in attrs.get('_name_list_private', [])})

        # return type.__new__(cls, name, bases, attrs)

    # @staticmethod
    # def generate_property_func_in_data(name):

        # def func(self):

            # if not hasattr(self, '_data'):
                # raise ValueError
            # else:
                # data = getattr(self, '_data')
                # if not hasattr(data, name):
                    # raise ValueError

            # return getattr(data, name)

        # return func

    # @staticmethod
    # def generate_property_func_private(name):

        # def func(self):

            # if not hasattr(self, f'_{name}'):
                # raise ValueError

            # return getattr(self, f'_{name}')

        # return func


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
        self.stock_pool = self.df_index_historical_constituents.loc[:, ['stock_id', 'stock_code']].drop_duplicates(subset=['stock_id']).set_index('stock_id').sort_index()

        df_merge_info = factor_ml_merge_list.load()
        self.df_merge_info = df_merge_info.loc[(df_merge_info.trade_date.isin(self.reindex)) & (df_merge_info.old_stock_id.isin(self.stock_pool.index))]

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

        ser_stock_company_id = caihui_tq_sk_basicinfo.load_stock_company_id(stock_ids=self.stock_pool.index).reset_index().set_index('company_id').stock_id

        df_stock_historical_share = caihui_tq_sk_sharestruchg.load_company_historical_share(
            company_ids=ser_stock_company_id.index,
            begin_date=self.reindex[0].strftime('%Y%m%d'),
            end_date=self.reindex[-1].strftime('%Y%m%d')
        )
        df_stock_historical_share['stock_id'] = df_stock_historical_share.company_id.map(ser_stock_company_id)
        df_stock_historical_share.drop('company_id', axis='columns', inplace=True)

        return df_stock_historical_share


class StockPortfolio: # (metaclass=MetaClassPropertyDecorator)

    _ref_list = {}

    # _name_list_in_data = [
        # 'index_id',
        # 'reindex',
        # 'look_back',
        # 'reindex_total',
        # 'df_index_historical_constituents',
        # 'stock_pool',
        # 'df_merge_info',
        # 'stock_pool_total',
        # 'df_stock_prc',
        # 'df_stock_ret',
        # 'df_stock_status',
        # 'df_stock_industry',
        # 'df_stock_historical_share',
        # 'stock_market_data',
        # 'stock_financial_data'
    # ]

    # _name_list_private = [
        # 'df_stock_pos',
        # 'df_stock_pos_adjusted',
        # 'ser_portfolio_nav',
        # 'ser_portfolio_inc',
        # 'ser_turnover'
    # ]

    def __init__(self, index_id, reindex, look_back, *args, **kwargs):

        ref = f'{index_id}, {reindex}, {look_back}'
        sha1 = hashlib.sha1()
        sha1.update(ref.encode('utf-8'))
        ref = sha1.hexdigest()

        if ref in StockPortfolio._ref_list:
            self._data = StockPortfolio._ref_list[ref]
        else:
            self._data = StockPortfolio._ref_list[ref] = StockPortfolioData(index_id, reindex, look_back)

        self._df_stock_pos = None
        self._df_stock_pos_adjusted = None

        self._ser_portfolio_nav = None
        self._ser_portfolio_inc = None
        self._ser_turnover = None

    @property
    def index_id(self):

        return self._data.index_id

    @property
    def reindex(self):

        return self._data.reindex

    @property
    def look_back(self):

        return self._data.look_back

    @property
    def reindex_total(self):

        return self._data.reindex_total

    @property
    def df_index_historical_constituents(self):

        return self._data.df_index_historical_constituents

    @property
    def stock_pool(self):

        return self._data.stock_pool

    @property
    def df_merge_info(self):

        return self._data.df_merge_info

    @property
    def stock_pool_total(self):

        return self._data.stock_pool_total

    @property
    def df_stock_prc(self):

        return self._data.df_stock_prc

    @property
    def df_stock_ret(self):

        return self._data.df_stock_ret

    @property
    def df_stock_status(self):

        return self._data.df_stock_status

    @property
    def df_stock_industry(self):

        return self._data.df_stock_industry

    @property
    def df_stock_historical_share(self):

        return self._data.df_stock_historical_share

    @property
    def stock_market_data(self):

        return self._data.stock_market_data

    @property
    def stock_financial_data(self):

        return self._data.stock_financial_data

    @property
    def ser_index_nav(self):

        return self._data.ser_stock_nav

    @property
    def df_stock_pos(self):

        return self._df_stock_pos

    @property
    def df_stock_pos_adjusted(self):

        return self._df_stock_pos_adjusted

    @property
    def ser_portfolio_nav(self):

        return self._ser_portfolio_nav

    @property
    def ser_portfolio_inc(self):

        return self._ser_portfolio_inc

    @property
    def ser_turnover(self):

        return self._ser_turnover

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
            stock_pos_adjusted.loc[index_adjustable_stock] = stock_pos_standard.loc[index_adjustable_stock] / sum_pos_standard * sum_pos_adjustable
            ser_turnover.loc[trade_date] = (stock_pos_adjusted - stock_pos).abs().sum()

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

    def __init__(self, index_id, reindex, look_back, *args, **kwargs):

        super(StockPortfolioMarketCap, self).__init__(index_id, reindex, look_back, *args, **kwargs)

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/000300
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        ser_stock_free_float_market_cap = self._calc_stock_free_float_market_cap(trade_date, stock_ids)

        stock_pos = pd.Series(index=stock_ids, name=trade_date)
        stock_pos.loc[:] = (ser_stock_free_float_market_cap).fillna(0.0)
        stock_pos.loc[:] = stock_pos / stock_pos.sum()

        return stock_pos

    def _calc_stock_free_float_market_cap(self, trade_date, stock_ids):

        df_stock_share = self._load_stock_share(trade_date, stock_ids)
        ser_stock_total_market_cap = self.stock_financial_data['total_market_cap'].loc[trade_date, stock_ids]

        ser_free_float_weight = df_stock_share.free_float_share / df_stock_share.total_share
        ser_free_float_weight.loc[:] = ser_free_float_weight.apply(self._weight_adjustment_algo)

        ser_stock_free_float_market_cap = ser_stock_total_market_cap * ser_free_float_weight

        return ser_stock_free_float_market_cap

    def _load_stock_share(self, trade_date, stock_ids):

        df_stock_share = self.df_stock_historical_share.loc[
            (self.df_stock_historical_share.begin_date<=trade_date) & \
            ((self.df_stock_historical_share.end_date>trade_date) | \
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

    def __init__(self, index_id, reindex, look_back, *args, **kwargs):

        super(StockPortfolioEqualWeight, self).__init__(index_id, reindex, look_back, *args, **kwargs)

    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        stock_pos = pd.Series(1.0/stock_ids.size, index=stock_ids, name=trade_date)

        return stock_pos


class StockPortfolioLowVolatility(StockPortfolio):

    def __init__(self, index_id, reindex, look_back, percentage, *args, **kwargs):

        super(StockPortfolioLowVolatility, self).__init__(index_id, reindex, look_back, *args, **kwargs)

        self._percentage = percentage

    @property
    def percentage(self):

        return self._percentage

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/000803
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        low_volatility_stock_ids, ser_low_volatility_stock_weight = self._calc_low_volatility(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[low_volatility_stock_ids] = ser_low_volatility_stock_weight.fillna(0.0)
        stock_pos.loc[:] = stock_pos / stock_pos.sum()

        return stock_pos

    def _calc_low_volatility(self, trade_date, stock_ids, percentage):

        df_stock_ret = self._load_stock_return(trade_date, stock_ids)
        df_stock_status = self.df_stock_status.loc[df_stock_ret.index, stock_ids]
        df_stock_ret[df_stock_status>2] = np.nan

        ser_stock_volatility = df_stock_ret.std()

        # Low volatility condition quantile as percentage
        portfolio_size = round(stock_ids.size * percentage)
        low_volatility_stock_ids = ser_stock_volatility.sort_values(ascending=True).iloc[:portfolio_size].index

        ser_low_volatility_stock_weight = 1.0 / ser_stock_volatility.loc[low_volatility_stock_ids]

        return low_volatility_stock_ids, ser_low_volatility_stock_weight


class StockPortfolioMomentum(StockPortfolioMarketCap):

    def __init__(self, index_id, reindex, look_back, percentage, exclusion, *args, **kwargs):

        super(StockPortfolioMomentum, self).__init__(index_id, reindex, look_back, *args, **kwargs)

        self._percentage = percentage
        self._exclusion = exclusion

    @property
    def percentage(self):

        return self._percentage

    @property
    def exclusion(self):

        return self._exclusion

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/H30260
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        momentum_stock_ids, _ = self._calc_momentum(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[momentum_stock_ids] = self._calc_stock_free_float_market_cap(trade_date, momentum_stock_ids).fillna(0.0)
        stock_pos.loc[:] = stock_pos / stock_pos.sum()

        return stock_pos

    def _calc_momentum(self, trade_date, stock_ids, percentage):

        df_stock_prc = self._load_stock_price(trade_date, stock_ids)

        ser_stock_momentum = df_stock_prc.iloc[-1-self.exclusion] / df_stock_prc.iloc[0]

        # Momentum condition quantile as percentage
        portfolio_size = round(stock_ids.size * percentage)
        momentum_stock_ids = ser_stock_momentum.sort_values(ascending=False).iloc[:portfolio_size].index

        ser_momentum_stock_weight = 1.0 / ser_stock_momentum.loc[momentum_stock_ids]

        return momentum_stock_ids, ser_momentum_stock_weight


class StockPortfolioSmallSize(StockPortfolioMarketCap):

    def __init__(self, index_id, reindex, look_back, percentage, *args, **kwargs):

        super(StockPortfolioSmallSize, self).__init__(index_id, reindex, look_back, *args, **kwargs)

        self._percentage = percentage

    @property
    def percentage(self):

        return self._percentage

    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        small_size_stock_ids, ser_small_size_stock_weight = self._calc_small_size(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[small_size_stock_ids] = ser_small_size_stock_weight.fillna(0.0)
        stock_pos.loc[:] = stock_pos / stock_pos.sum()

        return stock_pos

    def _calc_small_size(self, trade_date, stock_ids, percentage):

        ser_stock_free_float_market_cap = self._calc_stock_free_float_market_cap(trade_date, stock_ids)

        portfolio_size = round(stock_ids.size * self.percentage)
        small_size_stock_ids = ser_stock_free_float_market_cap.sort_values(ascending=True).iloc[:portfolio_size].index

        ser_small_size_stock_weight = ser_stock_free_float_market_cap.loc[small_size_stock_ids]

        return small_size_stock_ids, ser_small_size_stock_weight


class StockPortfolioLowBeta(StockPortfolio):

    def __init__(self, index_id, reindex, look_back, percentage, benchmark_id, *args, **kwargs):

        super(StockPortfolioLowBeta, self).__init__(index_id, reindex, look_back, percentage, benchmark_id, *args, **kwargs)

        self._percentage = percentage

        df_benchmark_nav = factor_sp_stock_portfolio_nav.load(benchmark_id)
        self._ser_benchmark_nav = df_benchmark_nav.nav.rename(benchmark_id)
        self._ser_benchmark_inc = df_benchmark_nav.inc.rename(benchmark_id)

    @property
    def percentage(self):

        return self._percentage

    @property
    def ser_benchmark_nav(self):

        return self._ser_benchmark_nav

    @property
    def ser_benchmark_inc(self):

        return self._ser_benchmark_inc

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/000829
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        low_beta_stock_ids, ser_low_beta_stock_weight = self._calc_low_beta(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[low_beta_stock_ids] = ser_low_beta_stock_weight.fillna(0.0)
        stock_pos.loc[:] = stock_pos / stock_pos.sum()

        return stock_pos

    def _calc_low_beta(self, trade_date, stock_ids, percentage):

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

        return low_beta_stock_ids, ser_low_beta_stock_weight

    def _calc_series_cov(self, ser1, ser2):

        df = pd.DataFrame({ser1.name: ser1, ser2.name: ser2})
        ser_cov = df.cov().iloc[0, 1]

        return ser_cov


class StockPortfolioLowBetaLowVolatility(StockPortfolioLowBeta, StockPortfolioLowVolatility):

    def __init__(self, index_id, reindex, look_back, percentage, benchmark_id, *args, **kwargs):

        super(StockPortfolioLowBetaLowVolatility, self).__init__(index_id, reindex, look_back, percentage, benchmark_id, *args, **kwargs)

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/930985
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        low_beta_stock_ids, _ = self._calc_low_beta(trade_date, stock_ids, self.percentage)
        low_beta_low_vol_stock_ids, _ = self._calc_low_volatility(trade_date, low_beta_stock_ids, 0.5)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[low_beta_stock_ids] = 1.0
        stock_pos.loc[:] = stock_pos / stock_pos.sum()

        return stock_pos


class StockPortfolioIndustry(StockPortfolio):

    def __init__(self, index_id, reindex, look_back, sw_industry_code, *args, **kwargs):

        super(StockPortfolioIndustry, self).__init__(index_id, reindex, look_back, *args, **kwargs)

        self._sw_industry_code = sw_industry_code

    @property
    def sw_industry_code(self):

        return self._sw_industry_code

    def _load_stock_pool(self, trade_date):

        set_trace()
        stock_pool = super(StockPortfolioIndustry, self)._load_stock_pool(trade_date)

        stock_ids_by_industry = self._load_stock_ids_by_industry()
        stock_pool = stock_pool.reindex(stock_ids_by_industry)

        return stock_pool

    def _load_stock_ids_by_industry(self):

        stock_ids_by_industry = self.df_stock_industry.loc[self.df_stock_industry.sw_level1_code==self.sw_industry_code].index

        return stock_ids_by_industry


class StockPortfolioIndustryLowVolatility(StockPortfolioIndustry, StockPortfolioLowVolatility):

    def __init__(self, index_id, reindex, look_back, sw_industry_code, *args, **kwargs):

        super(StockPortfolioIndustryLowVolatility, self).__init__(index_id, reindex, look_back, sw_industry_code, *args, **kwargs)


class StockPortfolioIndustryMomentum(StockPortfolioIndustry, StockPortfolioMomentum):

    def __init__(self, index_id, reindex, look_back, sw_industry_code, exclusion, *args, **kwargs):

        super(StockPortfolioIndustryMomentum, self).__init__(index_id, reindex, look_back, sw_industry_code, exclusion, *args, **kwargs)


def func(algo, index_id, trade_dates, look_back, sw_industry_code, *args, **kwargs):

    df = pd.DataFrame()

    class_name = f'StockPortfolio{algo}'
    cls = globals()[class_name]

    portfolio = cls(index_id, trade_dates, look_back, sw_industry_code, *args, **kwargs)
    df[sw_industry_code] = portfolio.calc_portfolio_nav()

    df.to_csv(f'{algo}_{sw_industry_code}.csv')

def multiprocessing_calc_portfolio_nav_by_industry(algo, index_id, trade_dates, look_back, *args, **kwargs):

    sw_industry_codes = caihui_tq_sk_basicinfo.load_sw_industry_code_info().index

    portfolio = StockPortfolio(index_id, trade_dates, look_back)

    for sw_industry_code in sw_industry_codes:

        process = multiprocessing.Process(target=func, args=(algo, index_id, trade_dates, look_back, sw_industry_code, *args), kwargs={**kwargs})
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

    # dict_portfolio['LowBetaLowVolatility'] = StockPortfolioLowBetaLowVolatility(index_id, trade_dates, look_back, percentage=0.6, benchmark_id='CS.000906')
    # df_portfolio_nav['LowBetaLowVolatility'] = dict_portfolio['LowBetaLowVolatility'].calc_portfolio_nav()

    # df_portfolio_nav.to_csv('df_portfolio_nav.csv')
    set_trace()

    # multiprocessing_calc_portfolio_nav_by_industry('IndustryLowVolatility', index_id, trade_dates, look_back, percentage=1.0)
    # set_trace()

