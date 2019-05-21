#coding=utf-8
'''
Created on: Mar. 11, 2019
Modified on: May. 17, 2019
Author: Shixun Su, Boyang Zhou, Ning Yang
Contact: sushixun@licaimofang.com
'''

import sys
import logging
from functools import partial
import multiprocessing
import numpy as np
import pandas as pd
import statsmodels.api as sm
from cvxopt import matrix, solvers
import math
import hashlib
import re
import copy
# from ipdb import set_trace
sys.path.append('shell')
from db import caihui_tq_ix_comp, caihui_tq_qt_index, caihui_tq_qt_skdailyprice, caihui_tq_sk_basicinfo, caihui_tq_sk_dquoteindic, caihui_tq_sk_finindic, caihui_tq_sk_sharestruchg
from db import wind_asharecalendar
from db import wind_aindexeodprices, wind_aindexmembers
from db import wind_asharedescription, wind_ashareeodprices
from db import wind_asharecapitalization, wind_asharefreefloat, wind_ashareeodderivativeindicator
from db import wind_ashareipo, wind_asharestockswap
from db import factor_financial_statement, factor_ml_merge_list, asset_sp_stock_portfolio_nav
from trade_date import ATradeDate
import calc_covariance
import calc_financial_descriptor
from util_timestamp import *
import statistic_tools_multifactor
from db import database
from config import *


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
                    raise AttributeError(f'\'{self.__class__.__name__}\'object has no attribute \'{variable}\'')

            return getattr(data, variable)

        return func

    @staticmethod
    def generate_func_for_instance_variable(variable):

        def func(self):

            if not hasattr(self, f'_{variable}'):
                raise AttributeError(f'\'{self.__class__.__name__}\'object has no attribute \'{variable}\'')

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

        self.ser_index_nav = wind_aindexeodprices.load_a_index_nav(
            index_ids=self.index_id,
            reindex=self.reindex_total,
            fill_method='pad'
        )[self.index_id]

        self.df_index_historical_constituents = wind_aindexmembers.load_a_index_historical_constituents(
            self.index_id,
            begin_date=self.reindex[0],
            end_date=self.reindex[-1]
        )

        self.stock_pool = wind_asharedescription.load_a_stock_code_info(
            stock_ids=self.df_index_historical_constituents.stock_id.unique()
        )

        self.df_stock_swap = wind_asharestockswap.load_a_stock_swap(
            transferer_stock_ids=self.stock_pool.index
        )

        self.stock_pool_total = wind_asharedescription.load_a_stock_code_info(
            stock_ids=self.stock_pool.index.union(self.df_stock_swap.targetcomp_stock_id)
        )

        self.df_stock_ipo = wind_ashareipo.load_a_stock_ipo_info(
            stock_ids=self.stock_pool.index
        )

        self.df_stock_status = wind_ashareeodprices.load_a_stock_status(
            stock_ids=self.stock_pool_total.index,
            reindex=self.reindex_total
        )

        self.df_stock_prc, self.df_stock_ret = self.__load_stock_price_and_return()

        self.df_stock_total_share, self.df_stock_free_float_share, self.df_stock_total_market_value = self.__load_stock_share_and_market_value()

        # self.stock_market_data = caihui_tq_sk_dquoteindic.load_stock_market_data(
            # stock_ids=self.stock_pool.index,
            # reindex=self.reindex_total,
            # fill_method='pad'
        # )

        # self.stock_financial_data = caihui_tq_sk_finindic.load_stock_financial_data(
            # stock_ids=self.stock_pool.index,
            # reindex=self.reindex_total,
            # fill_method='pad'
        # )

        self.df_stock_financial_descriptor = calc_financial_descriptor.calc_financial_descriptor(self.stock_pool.index)

    def __load_stock_price_and_return(self):

        df_stock_prc = wind_ashareeodprices.load_a_stock_adj_price(
            stock_ids=self.stock_pool_total.index,
            reindex=self.reindex_total,
            fill_method='pad'
        )

        for stock_id, stock_ipo in self.df_stock_ipo.iterrows():

            trade_date = trade_date_before(stock_ipo.ipo_date)
            if trade_date not in self.reindex:
                continue

            df_stock_prc.loc[trade_date, stock_id] = stock_ipo.ipo_price

        df_stock_ret = df_stock_prc.pct_change().iloc[1:]

        return df_stock_prc, df_stock_ret

    def __load_stock_share_and_market_value(self):

        df_stock_total_share = wind_asharecapitalization.load_a_stock_total_share(
            stock_ids=self.stock_pool.index
        )

        df_stock_free_float_share = wind_asharefreefloat.load_a_stock_free_float_share(
            stock_ids=self.stock_pool.index
        )

        df_stock_total_market_value = wind_ashareeodderivativeindicator.load_a_stock_total_market_value(
            stock_ids=self.stock_pool.index,
            reindex=self.reindex,
            fill_method='pad'
        )

        for stock_id, stock_ipo in self.df_stock_ipo.iterrows():

            trade_date = trade_date_before(stock_ipo.ipo_date)
            if trade_date not in self.reindex:
                continue

            df_stock_total_share.loc[(stock_id, trade_date), :] = df_stock_total_share.loc[(stock_id, stock_ipo.ipo_date)]
            try:
                df_stock_free_float_share.loc[(stock_id, trade_date), :] = df_stock_free_float_share.loc[(stock_id, stock_ipo.ipo_date)]
            except KeyError:
                pass
            # df_stock_free_float_share.loc[(stock_id, trade_date), :] = stock_ipo.ipo_amount
            df_stock_total_market_value.loc[trade_date, stock_id] = \
                df_stock_total_share.loc[(stock_id, trade_date), 'total_share'] * stock_ipo.ipo_price

        df_stock_total_share = df_stock_total_share.sort_index().reset_index()
        df_stock_total_share.loc[:, 'end_date'] = df_stock_total_share.begin_date.shift(-1)
        df_stock_total_share.loc[df_stock_total_share.stock_id!=df_stock_total_share.stock_id.shift(-1), 'end_date'] = np.nan

        df_stock_free_float_share = df_stock_free_float_share.sort_index().reset_index()
        df_stock_free_float_share.loc[:, 'end_date'] = df_stock_free_float_share.begin_date.shift(-1)
        df_stock_free_float_share.loc[df_stock_free_float_share.stock_id!=df_stock_free_float_share.stock_id.shift(-1), 'end_date'] = np.nan

        return df_stock_total_share, df_stock_free_float_share, df_stock_total_market_value


class StockPortfolio(metaclass=MetaClassPropertyFuncGenerater):

    _ref_list = {}

    _variable_list_in_data = [
        'index_id',
        'reindex',
        'look_back',
        'reindex_total',
        'ser_index_nav',
        'df_index_historical_constituents',
        'stock_pool',
        'df_stock_swap',
        'stock_pool_total',
        'df_stock_prc',
        'df_stock_ret',
        'df_stock_status',
        'df_stock_industry',
        'df_stock_total_share',
        'df_stock_free_float_share',
        'df_stock_total_market_value',
        'stock_market_data',
        'stock_financial_data',
        'df_stock_financial_descriptor'
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

        for key in self._kwargs_list:

            if key not in kwargs:
                raise TypeError(f'__init__() missing 1 required positional argument: \'{key}\'')
            value = kwargs.get(key)
            setattr(self, f'_{key}', value)
            setattr(self.__class__, key, property(MetaClassPropertyFuncGenerater.generate_func_for_instance_variable(key)))

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
                for _, stock_swap in self.df_stock_swap.loc[self.df_stock_swap.equity_registration_date==trade_date].iterrows():

                    pos = stock_pos_adjusted.loc[stock_swap.transferer_stock_id] \
                        * (stock_swap.conversion_ratio * stock_swap.targetcomp_conversion_prc / stock_swap.transferer_stock_prc)

                    stock_pos_adjusted.loc[stock_swap.targetcomp_stock_id] = pos
                    stock_pos_adjusted.loc[stock_swap.transferer_stock_id] = 0.0

            # df_stock_pos_adjusted.loc[trade_date] = stock_pos_adjusted
            arr_stock_pos_adjusted = np.append(arr_stock_pos_adjusted, stock_pos_adjusted.values.reshape(1, -1), axis=0)

        df_stock_pos_adjusted = pd.DataFrame(arr_stock_pos_adjusted, index=self.reindex, columns=self.stock_pool_total.index)

        # ser0_portfolio_inc = pd.Series(0.0, index=self.reindex, name='inc')
        # ser0_portfolio_nav = pd.Series(1.0, index=self.reindex, name='nav')
        # ser0_portfolio_inc.loc[:] = (df_stock_pos_adjusted.rename(index=trade_date_after) * (self.df_stock_prc / self.df_stock_prc.rename(index=trade_date_after) - 1.0)).sum(axis='columns')
        # ser0_portfolio_nav.loc[:] = (ser0_portfolio_inc + 1.0).cumprod()

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

    def _load_stock_price(self, trade_date, stock_ids, fillna=True):

        reindex = self.reindex_total[self.reindex_total<=trade_date][-self.look_back-1:]

        # df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_prc(stock_ids=stock_ids, reindex=reindex, fill_method='pad')
        df_stock_prc = self.df_stock_prc.reindex(index=reindex, columns=stock_ids)

        if not fillna:

            df_stock_status = self.df_stock_status.loc[reindex, stock_ids]
            df_stock_prc[df_stock_status>2] = np.nan

        return df_stock_prc

    def _load_stock_return(self, trade_date, stock_ids, fillna=True):

        reindex = self.reindex_total[self.reindex_total<=trade_date][-self.look_back-1:]

        # df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_prc(stock_ids=stock_ids, reindex=reindex, fill_method='pad')
        # df_stock_ret = df_stock_prc.pct_change().iloc[1:]
        df_stock_ret = self.df_stock_ret.reindex(index=reindex[1:], columns=stock_ids)

        if not fillna:

            df_stock_status = self.df_stock_status.loc[reindex[1:], stock_ids]
            df_stock_ret[df_stock_status>2] = np.nan

        return df_stock_ret

    def _load_stock_ids(self, trade_date):

        trade_date = trade_date_after(trade_date)

        # stock_pool = caihui_tq_ix_comp.load_index_constituents(self.index_id, date=trade_date.strftime('%Y%m%d'))
        stock_ids = self.df_index_historical_constituents.loc[
            (self.df_index_historical_constituents.in_date<=trade_date) & \
            ((self.df_index_historical_constituents.out_date>=trade_date) | \
            (self.df_index_historical_constituents.out_date.isna()))
        ].set_index('stock_id').index.sort_values()

        return stock_ids

    def _calc_portfolio_size(self, size, percentage, lower_bound=1):

        portfolio_size = round(size * percentage)
        portfolio_size = int(max(portfolio_size, lower_bound))

        return portfolio_size

    def portfolio_analysis(self, reindex=None):

        if self.ser_portfolio_nav is None:
            self.calc_portfolio_nav()

        if reindex is None:
            reindex = self.reindex

        portfolio_return = self.ser_portfolio_nav.loc[reindex[-1]] / self.ser_portfolio_nav.loc[reindex[0]] - 1.0
        ser_free_risk_rate = pd.Series(0.00013, index=reindex[1:])
        ser_excess_return = self._ser_portfolio_inc.reindex(reindex[1:]) - ser_free_risk_rate
        excess_return_std = ser_excess_return.std()
        sharpe_ratio = ser_excess_return.mean() / excess_return_std

        print(f'portfolio return: {portfolio_return}, sharpe ratio: {sharpe_ratio}.')

        max_drawdown_end_date = np.argmin(self.ser_portfolio_nav / np.maximum.accumulate(self.ser_portfolio_nav))
        max_drawdown_begin_date = np.argmax(self.ser_portfolio_nav[:max_drawdown_end_date])
        max_drawdown = self.ser_portfolio_nav[max_drawdown_end_date] / self.ser_portfolio_nav[max_drawdown_begin_date] - 1.0

        print(f'max drawdown: {max_drawdown}, begin date: {max_drawdown_begin_date}, end date: {max_drawdown_end_date}.')

        return portfolio_return, sharpe_ratio

    def portfolio_statistic(self, benchmark_id):

        if self.ser_portfolio_nav is None:
            self.calc_portfolio_nav()

        ser_benchmark_nav = asset_sp_stock_portfolio_nav.load(benchmark_id) \
            .nav.rename(benchmark_id)

        if ser_benchmark_nav.size == 0:

            print(f'Benchmark {benchmark_id} doesn\'t exist.')

            return

        # statistic_tools_multifactor.OLS_compare_summary(self.ser_portfolio_nav, ser_benchmark_nav)
        statistic_tools_multifactor.GLS_compare_summary(self.ser_portfolio_nav, ser_benchmark_nav)

        return


class StockPortfolioMarketValue(StockPortfolio):

    _kwargs_list = []

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioMarketValue, self).__init__(index_id, reindex, look_back, **kwargs)

    # Reference: http://www.csindex.com.cn/zh-CN/indices/index-detail/000300
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        _, ser_market_value_stock_weight = self._calc_market_value(trade_date, stock_ids)

        stock_pos = pd.Series(index=stock_ids, name=trade_date)
        stock_pos.loc[:] = (ser_market_value_stock_weight).fillna(0.0)

        return stock_pos

    def _calc_market_value(self, trade_date, stock_ids, **kwargs):

        ser_stock_free_float_market_value = self._calc_stock_free_float_market_value(trade_date, stock_ids)

        ser_market_value_stock_weight = ser_stock_free_float_market_value
        ser_market_value_stock_weight /= ser_market_value_stock_weight.sum()

        return stock_ids, ser_market_value_stock_weight

    def _calc_stock_free_float_market_value(self, trade_date, stock_ids):

        df_stock_share = self._load_stock_share(trade_date, stock_ids)
        ser_stock_total_market_value = self.df_stock_total_market_value.loc[trade_date, stock_ids]

        ser_free_float_weight = df_stock_share.free_float_share / df_stock_share.total_share
        ser_free_float_weight.loc[:] = ser_free_float_weight.apply(self._weight_adjustment_algo)

        ser_stock_free_float_market_value = (ser_stock_total_market_value * ser_free_float_weight).rename('free_float_market_value')

        return ser_stock_free_float_market_value

    def _load_stock_share(self, trade_date, stock_ids):

        df_stock_share = pd.DataFrame(index=stock_ids, columns=['total_share', 'free_float_share'])

        # get_data = partial(self._get_data, trade_date=trade_date)
        # df_stock_share.loc[:, 'total_share'] = self.df_stock_total_share.loc[stock_ids].total_share.groupby(level=0).apply(get_data)
        # df_stock_share.loc[:, 'free_float_share'] = self.df_stock_free_float_share.loc[stock_ids].free_float_share.groupby(level=0).apply(get_data)

        df_stock_share.loc[:, 'total_share'] = self.df_stock_total_share.loc[
            (self.df_stock_total_share.begin_date<=trade_date) & \
            ((self.df_stock_total_share.end_date>trade_date) | \
            (self.df_stock_total_share.end_date.isna()))
        ].set_index('stock_id').reindex(stock_ids).total_share

        df_stock_share.loc[:, 'free_float_share'] = self.df_stock_free_float_share.loc[
            (self.df_stock_free_float_share.begin_date<=trade_date) & \
            ((self.df_stock_free_float_share.end_date>trade_date) | \
            (self.df_stock_free_float_share.end_date.isna()))
        ].set_index('stock_id').reindex(stock_ids).free_float_share

        return df_stock_share

    def _get_data(self, ser, trade_date):

        ser = ser.loc[ser.name]
        data = ser.iloc[ser.index.get_loc(trade_date, method='pad')]

        return data

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

        stock_ids = self._load_stock_ids(trade_date)

        stock_pos = pd.Series(1.0/stock_ids.size, index=stock_ids, name=trade_date)

        return stock_pos


class StockPortfolioLowVolatility(StockPortfolio):

    _kwargs_list = [
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioLowVolatility, self).__init__(index_id, reindex, look_back, **kwargs)

    # Reference: http://www.csindex.com.cn/zh-CN/indices/index-detail/000803
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        low_volatility_stock_ids, ser_low_volatility_stock_weight = self._calc_low_volatility(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[low_volatility_stock_ids] = ser_low_volatility_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_low_volatility(self, trade_date, stock_ids, percentage, **kwargs):

        df_stock_ret = self._load_stock_return(trade_date, stock_ids, fillna=False)

        ser_stock_volatility = df_stock_ret.std()

        # Low volatility condition quantile as percentage
        portfolio_size = self._calc_portfolio_size(stock_ids.size, percentage)
        low_volatility_stock_ids = ser_stock_volatility.sort_values(ascending=True).iloc[:portfolio_size].index

        ser_low_volatility_stock_weight = 1.0 / ser_stock_volatility.loc[low_volatility_stock_ids]
        ser_low_volatility_stock_weight /= ser_low_volatility_stock_weight.sum()

        return low_volatility_stock_ids, ser_low_volatility_stock_weight


class StockPortfolioMomentum(StockPortfolioMarketValue):

    _kwargs_list = [
        'percentage',
        'exclusion'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioMomentum, self).__init__(index_id, reindex, look_back, **kwargs)

    # Reference: http://www.csindex.com.cn/zh-CN/indices/index-detail/H30260
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        momentum_stock_ids, _ = self._calc_momentum(trade_date, stock_ids, self.percentage)
        _, ser_market_cap_stock_weight = self._calc_market_cap(trade_date, momentum_stock_ids)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[momentum_stock_ids] = ser_market_cap_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_momentum(self, trade_date, stock_ids, percentage, **kwargs):

        df_stock_prc = self._load_stock_price(trade_date, stock_ids)
        df_stock_ret = self._load_stock_return(trade_date, stock_ids, fillna=False)

        ser_stock_momentum = (df_stock_prc.iloc[-1-self.exclusion] / df_stock_prc.iloc[0] - 1.0) \
            / df_stock_ret.std()

        # Momentum condition quantile as percentage
        portfolio_size = self._calc_portfolio_size(stock_ids.size, percentage)
        momentum_stock_ids = ser_stock_momentum.sort_values(ascending=False).iloc[:portfolio_size].index

        ser_momentum_stock_weight = ser_stock_momentum.loc[momentum_stock_ids]
        ser_momentum_stock_weight.loc[ser_momentum_stock_weight<0.0] = 0.0
        ser_momentum_stock_weight /= ser_momentum_stock_weight.sum()

        return momentum_stock_ids, ser_momentum_stock_weight


class StockPortfolioSmallSize(StockPortfolioMarketValue):

    _kwargs_list = [
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioSmallSize, self).__init__(index_id, reindex, look_back, **kwargs)

    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        small_size_stock_ids, ser_small_size_stock_weight = self._calc_small_size(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[small_size_stock_ids] = ser_small_size_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_small_size(self, trade_date, stock_ids, percentage, **kwargs):

        ser_stock_free_float_market_cap = self._calc_stock_free_float_market_cap(trade_date, stock_ids)

        portfolio_size = self._calc_portfolio_size(stock_ids.size, percentage)
        small_size_stock_ids = ser_stock_free_float_market_cap.sort_values(ascending=True).iloc[:portfolio_size].index

        # Small size condition quantile as percentage
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

        df_benchmark_nav = asset_sp_stock_portfolio_nav.load(self.benchmark_id)
        self._ser_benchmark_nav = df_benchmark_nav.nav.rename(self.benchmark_id)
        self._ser_benchmark_inc = df_benchmark_nav.inc.rename(self.benchmark_id)

    # Reference: http://www.csindex.com.cn/zh-CN/indices/index-detail/000829
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        low_beta_stock_ids, ser_low_beta_stock_weight = self._calc_low_beta(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[low_beta_stock_ids] = ser_low_beta_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_low_beta(self, trade_date, stock_ids, percentage, **kwargs):

        df_stock_ret = self._load_stock_return(trade_date, stock_ids, fillna=False)

        ser_benchmark_inc = self.ser_benchmark_inc.loc[df_stock_ret.index]
        benchmark_inc_std = ser_benchmark_inc.std()

        calc_stock_benchmark_cov = partial(self._calc_series_cov, ser_benchmark_inc)
        ser_stock_beta = df_stock_ret.apply(calc_stock_benchmark_cov) / (benchmark_inc_std ** 2)

        # Low beta condition quantile as percentage
        portfolio_size = self._calc_portfolio_size(stock_ids.size, percentage)
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

    # Reference: http://www.csindex.com.cn/zh-CN/indices/index-detail/000828
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        high_beta_stock_ids, ser_high_beta_stock_weight = self._calc_high_beta(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[high_beta_stock_ids] = ser_high_beta_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_high_beta(self, trade_date, stock_ids, percentage, **kwargs):

        df_stock_ret = self._load_stock_return(trade_date, stock_ids, fillna=False)

        ser_benchmark_inc = self.ser_benchmark_inc.loc[df_stock_ret.index]
        benchmark_inc_std = ser_benchmark_inc.std()

        calc_stock_benchmark_cov = partial(self._calc_series_cov, ser_benchmark_inc)
        ser_stock_beta = df_stock_ret.apply(calc_stock_benchmark_cov) / (benchmark_inc_std ** 2)

        # High beta condition quantile as percentage
        portfolio_size = self._calc_portfolio_size(stock_ids.size, percentage)
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

        stock_ids = self._load_stock_ids(trade_date)

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

    # Reference: http://www.csindex.com.cn/zh-CN/indices/index-detail/930985
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        low_beta_stock_ids, _ = self._calc_low_beta(trade_date, stock_ids, self.percentage_low_beta)
        low_beta_low_volatility_stock_ids, _ = self._calc_low_volatility(trade_date, low_beta_stock_ids, self.percentage_low_volatility)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[low_beta_low_volatility_stock_ids] = 1.0 / low_beta_low_volatility_stock_ids.size

        return stock_pos


class StockPortfolioSectorNeutral(StockPortfolioMarketValue):

    _kwargs_list = [
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioSectorNeutral, self).__init__(index_id, reindex, look_back, **kwargs)

    # Reference: http://www.csindex.com.cn/zh-CN/indices/index-detail/930846
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        sector_neutral_stock_ids, ser_sector_neutral_stock_weight = self._calc_sector_neutral(trade_date, stock_ids, self.percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[sector_neutral_stock_ids] = ser_sector_neutral_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_sector_neutral(self, trade_date, stock_ids, percentage, **kwargs):

        ser_industry_free_float_market_cap = self._calc_industry_free_float_market_cap(trade_date, stock_ids)

        sector_neutral_stock_ids = pd.Index([], name='stock_id')
        ser_sector_neutral_stock_weight = pd.Series(index=stock_ids, name=trade_date)

        ser_percentage_by_industry = self._calc_percentage_by_industry(stock_ids, percentage)

        for sw_industry_code in ser_industry_free_float_market_cap.index:

            sw_industry_stock_ids = stock_ids.intersection(self.df_stock_industry.index[self.df_stock_industry.sw_level1_code==sw_industry_code])

            class_name = self.__class__.__name__
            algo = re.sub('StockPortfolioSectorNeutral', '', class_name, count=1)
            func_name = '_calc' + re.sub('(?P<l>[A-Z]+)', lambda x: '_'+x.group('l').lower(), algo)

            if not hasattr(self, func_name):
                raise AttributeError(f'\'{class_name}\'object has no attribute \'{func_name}\'')
            calc_func = getattr(self, func_name)
            stock_ids_by_industry, ser_stock_weight_by_industry = calc_func(
                trade_date,
                sw_industry_stock_ids,
                ser_percentage_by_industry.loc[sw_industry_code]
            )

            sector_neutral_stock_ids = sector_neutral_stock_ids.append(stock_ids_by_industry)
            ser_sector_neutral_stock_weight.loc[stock_ids_by_industry] = ser_stock_weight_by_industry \
                * ser_industry_free_float_market_cap.loc[sw_industry_code]

        sector_neutral_stock_ids = sector_neutral_stock_ids.sort_values()
        ser_sector_neutral_stock_weight = ser_sector_neutral_stock_weight.loc[sector_neutral_stock_ids]
        ser_sector_neutral_stock_weight.loc[:] /= ser_sector_neutral_stock_weight.sum()

        return sector_neutral_stock_ids, ser_sector_neutral_stock_weight

    def _calc_industry_free_float_market_cap(self, trade_date, stock_ids, **kwargs):

        ser_stock_free_float_market_cap = self._calc_stock_free_float_market_cap(trade_date, stock_ids)
        df_stock_industry = self.df_stock_industry.loc[stock_ids]

        ser_industry_free_float_market_cap = ser_stock_free_float_market_cap \
            .rename(index=df_stock_industry.sw_level1_code).rename_axis('sw_industry_code').groupby(by='sw_industry_code').sum()

        return ser_industry_free_float_market_cap

    def _calc_percentage_by_industry(self, stock_ids, percentage):

        portfolio_size = self._calc_portfolio_size(stock_ids.size, percentage)

        ser_size = self.df_stock_industry.loc[stock_ids].groupby(['sw_level1_code']).size()
        ser_portfolio_size = ser_size.apply(partial(self._calc_portfolio_size, percentage=percentage))

        delta_size = ser_portfolio_size.sum() - portfolio_size
        ser_portfolio_size.loc[ser_portfolio_size.sort_values(ascending=False).iloc[:abs(delta_size)].index] -= np.sign(delta_size)

        ser_percentage = (ser_portfolio_size / ser_size).rename('percentage')

        return ser_percentage


class StockPortfolioSectorNeutralLowVolatility(StockPortfolioSectorNeutral, StockPortfolioLowVolatility):

    _kwargs_list = [
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioSectorNeutralLowVolatility, self).__init__(index_id, reindex, look_back, **kwargs)


class StockPortfolioSectorNeutralLowBeta(StockPortfolioSectorNeutral, StockPortfolioLowBeta):

    _kwargs_list = [
        'percentage',
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

        sw_industry_stock_ids = self._load_stock_ids(trade_date)

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

    def _load_stock_ids(self, trade_date):

        stock_ids = super(StockPortfolioIndustry, self)._load_stock_ids(trade_date)
        '''bug'''
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


class StockPortfolioFamaMacbethRegression(StockPortfolio):

    _kwargs_list = [
        'trading_frequency',
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioFamaMacbethRegression, self).__init__(index_id, reindex, look_back, **kwargs)

        self.finished_regression = self._load_factor_return_daily()
        self.factor_return_daily = self._fama_macbeth_regression_daily()
        self.reindex_regression = list(self.factor_return_daily.index)
        self.residual_return = self._calc_residual_return()

    def _calc_stock_pos(self, trade_date):
        # calculate factor weight
        all_style = ['VALUE', 'QUALITY', 'GROWTH', 'LSIZE', 'LIQUIDITY', 'BETA', 'STREV']
        # 调仓频率
        trade_date_num = list(self.reindex).index(trade_date)
        calc_pos_num = (trade_date_num // self.trading_frequency) * self.trading_frequency
        calc_pos_date = self.reindex[calc_pos_num]
        if calc_pos_date not in self.reindex_regression:
            optimal_weight_t = pd.Series(1 / len(all_style), index=all_style)
        else:
            regression_date_num = self.reindex_regression.index(calc_pos_date)
            if regression_date_num < self.trading_frequency * 50:
                optimal_weight_t = pd.Series(1/len(all_style), index=all_style)
            else:
                residual_return_t = self.residual_return.loc[:calc_pos_date, all_style].iloc[:-1].iloc[::-self.trading_frequency].sort_index().copy()
                covariance_t = calc_covariance.calc_covariance(data=residual_return_t, lookback_period=128, H_L_vol=32, Lags_vol=2, H_L_corr=64, Lags_corr=2, Predict_period=1)
                P_t = matrix(covariance_t)
                q_t = matrix(np.zeros((covariance_t.shape[0], 1)))
                G_t = matrix(np.eye(covariance_t.shape[0]) * -1)
                h_t = matrix(np.zeros(covariance_t.shape[0]))
                A_t = matrix(np.ones(covariance_t.shape[0]).reshape(1, -1))
                b_t = matrix([1.0])
                solvers.options['show_progress'] = False  # Notice
                sol = solvers.qp(P=P_t, q=q_t, G=G_t, h=h_t, A=A_t, b=b_t)
                opt_t = np.array(sol['x']).T[0]
                optimal_weight_t = pd.Series(opt_t, index=all_style)
        # calc_pos_date 为保证交易的可行性，使用前一天的因子暴露来计算今天收盘时的股票仓位
        last_trade_num = list(self.reindex_total).index(calc_pos_date) - 1
        last_trade_date = list(self.reindex_total)[last_trade_num]

        stock_ids = self._load_stock_ids(trade_date)
        data_FS_t = self._select_report_period(last_trade_date=last_trade_date, stock_ids=stock_ids, select_method='radical')
        data_value_t = self._calc_value_descriptor(last_trade_date=last_trade_date, stock_ids=stock_ids)
        data_numerical_t = self._calc_numerical_descriptor(last_trade_date=last_trade_date, stock_ids=stock_ids)
        data_industry_t = self.df_stock_industry[['sw_level1_name']].reindex(stock_ids).rename(columns={'sw_level1_name': 'INDUSTRY'}).copy()
        data_descriptor_t = pd.concat([data_FS_t, data_value_t, data_numerical_t, data_industry_t], axis=1, join='outer', sort=False)
        data_descriptor_t['LSIZE'] = np.log(data_descriptor_t.negotiable_market_value)
        financial_descriptor = ['pe_ttm', 'pb', 'ps_ttm', 'pc_ttm', 'dy', 'evebitda', 'QUALITY_ROA', 'QUALITY_ROE',
                                'QUALITY_ACCF', 'QUALITY_GPM', 'QUALITY_VERN', 'QUALITY_AGRO', 'GROWTH_EGRO',
                                'GROWTH_CGRO', 'GROWTH_GPGRO', 'GROWTH_GPMGRO', 'GROWTH_ATOGRO']
        liquidity_descriptor = ['STOM', 'STOQ', 'STOA']
        all_descriptor = financial_descriptor + liquidity_descriptor
        data_descriptor_t = self._z_score_cbi(data=data_descriptor_t, columns=financial_descriptor, industry_column='INDUSTRY')
        data_descriptor_t = self._z_score(data=data_descriptor_t, columns=all_descriptor)
        data_descriptor_t['VALUE'] = data_descriptor_t[['pe_ttm', 'pb', 'ps_ttm', 'pc_ttm', 'dy', 'evebitda']].mean(axis=1, skipna=True)
        data_descriptor_t['QUALITY'] = data_descriptor_t[['QUALITY_ROA', 'QUALITY_ROE', 'QUALITY_ACCF', 'QUALITY_GPM', 'QUALITY_VERN', 'QUALITY_AGRO']].mean(axis=1, skipna=True)
        data_descriptor_t['GROWTH'] = data_descriptor_t[['GROWTH_EGRO', 'GROWTH_CGRO', 'GROWTH_GPGRO', 'GROWTH_GPMGRO', 'GROWTH_ATOGRO']].mean(axis=1, skipna=True)
        data_descriptor_t['LIQUIDITY'] = data_descriptor_t[['STOM', 'STOQ', 'STOA']].mean(axis=1, skipna=True)

        all_style = ['VALUE', 'QUALITY', 'GROWTH', 'LSIZE', 'LIQUIDITY', 'BETA', 'STREV']
        data_descriptor_t = self._z_score(data=data_descriptor_t, columns=all_style)
        data_descriptor_t[['LSIZE', 'LIQUIDITY', 'STREV']] = - data_descriptor_t[['LSIZE', 'LIQUIDITY', 'STREV']]
        data_descriptor_t[all_style] = data_descriptor_t[all_style].fillna(0.0)
        data_descriptor_t['SCORE'] = 0.0
        for j_num, j_style in enumerate(all_style):
            data_descriptor_t['SCORE'] = data_descriptor_t['SCORE'] + optimal_weight_t[j_style] * data_descriptor_t[j_style]
        data_descriptor_t.sort_values(by='SCORE', ascending=False, inplace=True)
        select_num_t = int(data_descriptor_t.shape[0] * self.percentage)
        stock_pos = pd.Series(1.0/select_num_t, index=data_descriptor_t.iloc[:select_num_t].index, name=trade_date)
        return stock_pos

    def _load_factor_return_daily(self):
        conn = pymysql.connect(host=db_multi_factor['host'], user=db_multi_factor['user'], passwd=db_multi_factor['passwd'], database='multi_factor', charset='utf8')
        sql_table = 'SELECT table_name FROM information_schema.TABLES WHERE table_name = "factor_return_daily"'
        if conn.cursor().execute(sql_table):  # 存在表
            factor_return_daily = pd.read_sql(sql='select * from factor_return_daily', con=conn, parse_dates=['trade_date'])
            finished_regression = list(factor_return_daily.trade_date)
        else:  # 不存在表
            finished_regression = []
        conn.close()
        return finished_regression

    def _fama_macbeth_regression_daily(self):
        # notice trading_frequency
        factor_return_daily = pd.DataFrame()
        for last_trade_date, trade_date in zip(self.reindex[:-self.trading_frequency], self.reindex[self.trading_frequency:]):
            if trade_date in self.finished_regression:
                continue
            stock_ids = self._load_stock_ids(last_trade_date)
            data_FS_t = self._select_report_period(last_trade_date=last_trade_date, stock_ids=stock_ids, select_method='radical')
            data_value_t = self._calc_value_descriptor(last_trade_date=last_trade_date, stock_ids=stock_ids)
            data_numerical_t = self._calc_numerical_descriptor(last_trade_date=last_trade_date, stock_ids=stock_ids)
            data_industry_t = self.df_stock_industry[['sw_level1_name']].reindex(stock_ids).rename(columns={'sw_level1_name': 'INDUSTRY'}).copy()
            data_price_t = self.df_stock_prc.loc[[last_trade_date, trade_date], list(stock_ids)].copy()
            data_return_t = data_price_t.pct_change().iloc[1]
            data_return_t.name = 'NEXT_RETURN'
            data_return_t = pd.DataFrame(data_return_t)  # check that
            data_descriptor_t = pd.concat([data_FS_t, data_value_t, data_numerical_t, data_industry_t, data_return_t], axis=1, join='outer', sort=False)
            data_descriptor_t['LSIZE'] = np.log(data_descriptor_t.negotiable_market_value)
            financial_descriptor = ['pe_ttm', 'pb', 'ps_ttm', 'pc_ttm', 'dy', 'evebitda', 'QUALITY_ROA', 'QUALITY_ROE',
                                    'QUALITY_ACCF', 'QUALITY_GPM', 'QUALITY_VERN', 'QUALITY_AGRO', 'GROWTH_EGRO',
                                    'GROWTH_CGRO', 'GROWTH_GPGRO', 'GROWTH_GPMGRO', 'GROWTH_ATOGRO']
            liquidity_descriptor = ['STOM', 'STOQ', 'STOA']
            all_descriptor = financial_descriptor + liquidity_descriptor
            data_descriptor_t = self._z_score_cbi(data=data_descriptor_t, columns=financial_descriptor, industry_column='INDUSTRY')
            data_descriptor_t = self._z_score(data=data_descriptor_t, columns=all_descriptor)
            data_descriptor_t['VALUE'] = data_descriptor_t[['pe_ttm', 'pb', 'ps_ttm', 'pc_ttm', 'dy', 'evebitda']].mean(axis=1, skipna=True)
            data_descriptor_t['QUALITY'] = data_descriptor_t[['QUALITY_ROA', 'QUALITY_ROE', 'QUALITY_ACCF', 'QUALITY_GPM', 'QUALITY_VERN', 'QUALITY_AGRO']].mean(axis=1, skipna=True)
            data_descriptor_t['GROWTH'] = data_descriptor_t[['GROWTH_EGRO', 'GROWTH_CGRO', 'GROWTH_GPGRO', 'GROWTH_GPMGRO', 'GROWTH_ATOGRO']].mean(axis=1, skipna=True)
            data_descriptor_t['LIQUIDITY'] = data_descriptor_t[['STOM', 'STOQ', 'STOA']].mean(axis=1, skipna=True)

            all_style = ['VALUE', 'QUALITY', 'GROWTH', 'LSIZE', 'LIQUIDITY', 'BETA', 'STREV']
            data_descriptor_t = self._z_score(data=data_descriptor_t, columns=all_style)
            data_descriptor_t[['LSIZE', 'LIQUIDITY', 'STREV']] = - data_descriptor_t[['LSIZE', 'LIQUIDITY', 'STREV']]
            data_descriptor_t[all_style] = data_descriptor_t[all_style].fillna(0.0)
            data_descriptor_t.dropna(subset=['NEXT_RETURN'], inplace=True)
            # OLS
            data_dummies = pd.get_dummies(data_descriptor_t.INDUSTRY)
            industry_t = list(data_dummies.columns)
            data_descriptor_t = pd.merge(data_dummies, data_descriptor_t, left_index=True, right_index=True, sort=False)
            factor_t = industry_t + all_style
            X = data_descriptor_t[factor_t].values
            Y = data_descriptor_t.NEXT_RETURN.values
            ols_results = sm.OLS(Y, X).fit()
            factor_return_dict = dict(zip(['trade_date']+factor_t, [trade_date]+list(ols_results.params)))
            factor_return_daily = factor_return_daily.append(pd.DataFrame(factor_return_dict, index=[0]), ignore_index=True, sort=False)
        # 写入数据库
        if not factor_return_daily.empty:
            factor_return_daily.trade_date = pd.to_datetime(factor_return_daily.trade_date).astype(pd.Timestamp)
            factor_return_daily = factor_return_daily[all_style+['trade_date']]
        conn = database.connection('factor')
        if len(self.finished_regression) != 0:
            factor_return_daily_exist = pd.read_sql(sql='select * from factor_return_daily', con=conn, parse_dates=['trade_date'])
            factor_return_daily = factor_return_daily.append(factor_return_daily_exist, ignore_index=True, sort=False)
        factor_return_daily = factor_return_daily.drop_duplicates(subset=['trade_date'], keep='first').sort_values(by=['trade_date'])
        factor_return_daily.trade_date = factor_return_daily.trade_date.map(lambda x: pd.Timestamp.strftime(x, '%Y-%m-%d'))
        pd.io.sql.to_sql(factor_return_daily, 'factor_return_daily', con=conn, if_exists='replace', index=False)
        factor_return_daily.trade_date = pd.to_datetime(factor_return_daily.trade_date).astype(pd.Timestamp)
        return factor_return_daily.set_index('trade_date')

    def _calc_residual_return(self):
        residual_return = pd.DataFrame()
        for last_trade_date, trade_date in zip(self.reindex_regression[:-self.trading_frequency], self.reindex_regression[self.trading_frequency:]):
            expected_return_t = self.factor_return_daily.loc[:last_trade_date].mean(axis=0)
            real_return_t = self.factor_return_daily.loc[trade_date]
            residual_return_t = pd.DataFrame(real_return_t - expected_return_t).T
            residual_return_t['trade_date'] = trade_date
            residual_return = residual_return.append(residual_return_t, ignore_index=True)
        return residual_return.set_index('trade_date')

    def _select_report_period(self, last_trade_date, stock_ids, select_method='radical'):
        df_stock_financial_descriptor = self.df_stock_financial_descriptor.copy()
        if select_method == 'radical':
            data_FS = df_stock_financial_descriptor.loc[df_stock_financial_descriptor.ACTUAL_ANN_DT <= last_trade_date].copy()
            data_FS = data_FS.sort_values(by=['WIND_CODE', 'ACTUAL_ANN_DT'], ascending=False).drop_duplicates(subset=['WIND_CODE'], keep='first')
        else:
            if last_trade_date.month < 5:
                report_period_t = pd.Timestamp(last_trade_date.year - 1, 9, 30)
            elif last_trade_date.month < 9:
                report_period_t = pd.Timestamp(last_trade_date.year, 3, 31)
            elif last_trade_date.month < 11:
                report_period_t = pd.Timestamp(last_trade_date.year, 6, 30)
            else:
                report_period_t = pd.Timestamp(last_trade_date.year, 9, 30)
            data_FS = df_stock_financial_descriptor.loc[df_stock_financial_descriptor.REPORT_PERIOD == report_period_t].copy()
        data_FS.set_index('stock_id', inplace=True)
        return data_FS.reindex(stock_ids)

    def _calc_value_descriptor(self, last_trade_date, stock_ids):
        data_financial = self.stock_financial_data.loc[last_trade_date].swaplevel(i=-2, j=-1).unstack().copy()
        data_financial = data_financial.astype(np.float)
        columns_retain = ['negotiable_market_value', 'pe_ttm', 'pb', 'ps_ttm', 'pc_ttm', 'dy', 'evebitda']
        data_financial = data_financial[columns_retain]
        data_financial[['pe_ttm', 'pb', 'ps_ttm', 'pc_ttm', 'evebitda']] = 1.0 / data_financial[['pe_ttm', 'pb', 'ps_ttm', 'pc_ttm', 'evebitda']]
        data_financial[['dy']] = data_financial[['dy']].fillna(0.0)
        return data_financial.reindex(stock_ids)

    def _calc_numerical_descriptor(self, last_trade_date, stock_ids):
        # beta
        def calc_beta(stock, benchmark):
            weight_t = (0.5 ** (1 / 63)) ** (np.arange(len(stock) - 1, -1, -1))
            y = stock * weight_t
            x = benchmark * weight_t
            beta = sm.OLS(y, sm.add_constant(x)).fit().params[1]
            return beta

        def calc_rs(data):
            weight_t = (0.5 ** (1 / 63)) ** (np.arange(len(data) - 1, -1, -1))
            relative_strength = np.dot(data, weight_t)
            return relative_strength

        df_stock_ret = self._load_stock_return(last_trade_date, stock_ids, fillna=True).fillna(0.0)
        df_stock_ret['benchmark'] = df_stock_ret.mean(axis=1)

        df_beta = df_stock_ret[stock_ids].apply(lambda x: calc_beta(x, df_stock_ret.benchmark))
        df_beta = pd.DataFrame(df_beta, columns=['BETA'])

        df_rs = df_stock_ret.sub(df_stock_ret.benchmark, axis=0)[stock_ids].rolling(63).apply(lambda x: calc_rs(x), 'raw=True')
        df_strev = pd.DataFrame(df_rs.iloc[-3:].mean(), columns=['STREV'])

        df_turnover = self.stock_market_data.loc[:last_trade_date].iloc[-256:].volume.copy() / self.stock_market_data.loc[:last_trade_date].iloc[-256:].market_share.copy() / 100
        df_stom = pd.DataFrame(np.log(df_turnover.iloc[-21:].mean() * 21), columns=['STOM'])
        df_stoq = pd.DataFrame(np.log(df_turnover.iloc[-63:].mean() * 21), columns=['STOQ'])
        df_stoa = pd.DataFrame(np.log(df_turnover.iloc[-256:].mean() * 21), columns=['STOA'])

        df_numerical_descriptor = pd.concat([df_beta, df_strev, df_stom, df_stoq, df_stoa], axis=1, join='outer', sort=False)
        return df_numerical_descriptor.reindex(stock_ids)

    def _z_score(self, data, columns):
        df = data.copy()
        for i_column in columns:
            loc_t = df[i_column].isin([np.nan, - np.inf, np.inf])
            df.loc[loc_t, i_column] = np.nan
            df.loc[df[i_column] > df[i_column].mean(skipna=True) + 3 * df[i_column].std(skipna=True), i_column] = df[i_column].mean(skipna=True) + 3 * df[i_column].std(skipna=True)
            df.loc[df[i_column] < df[i_column].mean(skipna=True) - 3 * df[i_column].std(skipna=True), i_column] = df[i_column].mean(skipna=True) - 3 * df[i_column].std(skipna=True)
            df[i_column] = (df[i_column] - df[i_column].mean(skipna=True)) / df[i_column].std(skipna=True)
        return df

    def _z_score_cbi(self, data, columns, industry_column='sw_level1_name'):
        df = data.copy()
        df.dropna(subset=[industry_column], inplace=True)
        industry_t = list(df[industry_column].unique())
        df_filter = pd.DataFrame()
        for i_industry in industry_t:
            loc_t = df[industry_column] == i_industry
            df_t = df.loc[loc_t].copy()
            if len(df_t.shape) == 2:
                if df_t.shape[0] >= 15:
                    df_t = self._z_score(data=df_t, columns=columns)
                    df_filter = df_filter.append(df_t, ignore_index=False)
        return df_filter


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

    index_id = '000906.SH'
    begin_date = '2019-03-04'
    end_date = '2019-03-29'
    look_back = 244

    trade_dates = ATradeDate.trade_date(begin_date=begin_date, end_date=end_date).rename('trade_date')

    dict_portfolio = {}
    df_portfolio_nav = pd.DataFrame()

    # dict_portfolio['MarketCap'] = StockPortfolioMarketValue(index_id, trade_dates, look_back)
    # df_portfolio_nav['MarketCap'] = dict_portfolio['MarketCap'].calc_portfolio_nav()

    # dict_portfolio['EqualWeight'] = StockPortfolioEqualWeight(index_id, trade_dates, look_back)
    # df_portfolio_nav['EqualWeight'] = dict_portfolio['EqualWeight'].calc_portfolio_nav()

    # dict_portfolio['LowVolatility'] = StockPortfolioLowVolatility(index_id, trade_dates, look_back, percentage=0.3)
    # df_portfolio_nav['LowVolatility'] = dict_portfolio['LowVolatility'].calc_portfolio_nav()
    # dict_portfolio['LowVolatility'].portfolio_analysis()
    # dict_portfolio['LowVolatility'].portfolio_statistic('CS.000906')

    # dict_portfolio['Momentum'] = StockPortfolioMomentum(index_id, trade_dates, look_back, percentage=0.3, exclusion=20)
    # tdf_portfolio_nav['Momentum'] = dict_portfolio['Momentum'].calc_portfolio_nav()

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

    # dict_portfolio['FamaMacbethRegression'] = StockPortfolioFamaMacbethRegression(index_id, trade_dates, look_back, trading_frequency=10, percentage=0.15)
    # df_portfolio_nav['FamaMacbethRegression'] = dict_portfolio['FamaMacbethRegression'].calc_portfolio_nav()

    # df_portfolio_nav.to_csv('df_portfolio_nav.csv')
    # set_trace()

    # multiprocessing_calc_portfolio_nav_by_industry('IndustryLowVolatility', index_id, trade_dates, look_back, percentage=0.30)

    # multiprocessing_calc_portfolio_nav_by_industry('IndustryMomentum', index_id, trade_dates, look_back, percentage=0.3, exclusion=30)
    # set_trace()

