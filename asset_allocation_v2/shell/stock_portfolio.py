#coding=utf-8
'''
Created on: Mar. 11, 2019
Modified on: Jun. 12, 2019
Author: Shixun Su, Boyang Zhou, Ning Yang
Contact: sushixun@licaimofang.com
'''

import sys
import logging
from functools import partial
import multiprocessing
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hierarchy
import statsmodels.api as sm
import cvxopt
from cvxopt import matrix, solvers
import math
import hashlib
import re
import copy
from ipdb import set_trace
sys.path.append('shell')
from db import caihui_tq_sk_basicinfo, caihui_tq_sk_dquoteindic, caihui_tq_sk_finindic
from db import wind_asharecalendar
from db import wind_aindexeodprices, wind_aindexmembers
from db import wind_asharedescription, wind_ashareswindustriesclass, wind_ashareeodprices
from db import wind_asharecapitalization, wind_asharefreefloat, wind_ashareeodderivativeindicator, wind_asharedividend
from db import wind_ashareipo, wind_asharestockswap
from db import factor_financial_statement
from db import asset_sp_stock_portfolio, asset_sp_stock_portfolio_nav, asset_sp_stock_portfolio_pos
from trade_date import ATradeDate
import calc_covariance
import calc_descriptor
from util_timestamp import *
import statistic_tools_multifactor
from db import database
from config import *
from db import factor_sf_stock_factor_exposure
import statsmodels.api as sm


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

        self.trade_dates = ATradeDate.trade_date(
            begin_date=self.reindex[0],
            end_date=self.reindex[-1]
        ).rename('trade_date')

        self.trade_dates_total = ATradeDate.trade_date(
            begin_date=self.reindex[0],
            end_date=self.reindex[-1],
            lookback=self.look_back+1
        ).rename('trade_date')

        self.ser_index_nav = wind_aindexeodprices.load_a_index_nav(
            index_ids=self.index_id,
            reindex=self.trade_dates_total,
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
        self.df_stock_swap.dropna(inplace=True)

        self.stock_pool_total = wind_asharedescription.load_a_stock_code_info(
            stock_ids=self.stock_pool.index.union(self.df_stock_swap.targetcomp_stock_id)
        )

        self.df_stock_ipo = wind_ashareipo.load_a_stock_ipo_info(
            stock_ids=self.stock_pool.index
        )

        self.df_stock_status = wind_ashareeodprices.load_a_stock_status(
            stock_ids=self.stock_pool_total.index,
            reindex=self.trade_dates_total
        )

        self.df_stock_historical_industry = wind_ashareswindustriesclass.load_a_stock_historical_sw_industry(
            stock_ids=self.stock_pool.index
        )

        self.df_stock_prc, self.df_stock_ret = self._load_stock_price_and_return()

        self.df_stock_total_share, self.df_stock_free_float_share, self.df_stock_total_market_value = self._load_stock_share_and_market_value()

        self.df_dividend = wind_asharedividend.load_a_stock_dividend(
            stock_ids=self.stock_pool.index
        )

        self.df_pdps_ttm = wind_ashareeodderivativeindicator.load_a_stock_derivative_indicator(
            stock_ids=self.stock_pool.index
        ).pdps_ttm

        # self.stock_market_data = caihui_tq_sk_dquoteindic.load_stock_market_data(
            # stock_ids=self.stock_pool.index,
            # reindex=self.trade_dates_total,
            # fill_method='pad'
        # )

        # self.stock_financial_data = caihui_tq_sk_finindic.load_stock_financial_data(
            # stock_ids=self.stock_pool.index,
            # reindex=self.trade_dates_total,
            # fill_method='pad'
        # )

        # self.df_stock_financial_descriptor = calc_descriptor.calc_stock_financial_descriptor(self.stock_pool.index)

    def _load_stock_price_and_return(self):

        df_stock_prc = wind_ashareeodprices.load_a_stock_adj_price(
            stock_ids=self.stock_pool_total.index,
            reindex=self.trade_dates_total,
            fill_method='pad'
        )

        for stock_id, stock_ipo in self.df_stock_ipo.iterrows():

            trade_date = trade_date_before(stock_ipo.ipo_date)
            if trade_date not in self.trade_dates_total:
                continue

            df_stock_prc.loc[trade_date, stock_id] = stock_ipo.ipo_price

        df_stock_ret = df_stock_prc.pct_change().iloc[1:]

        return df_stock_prc, df_stock_ret

    def _load_stock_share_and_market_value(self):

        df_stock_total_share = wind_asharecapitalization.load_a_stock_total_share(
            stock_ids=self.stock_pool.index
        )

        df_stock_free_float_share = wind_asharefreefloat.load_a_stock_free_float_share(
            stock_ids=self.stock_pool.index
        )

        df_stock_total_market_value = wind_ashareeodderivativeindicator.load_a_stock_total_market_value(
            stock_ids=self.stock_pool.index,
            reindex=self.trade_dates_total,
            fill_method='pad'
        )

        for stock_id, stock_ipo in self.df_stock_ipo.iterrows():

            trade_date = trade_date_before(stock_ipo.ipo_date)
            if trade_date not in self.trade_dates_total:
                continue

            try:
                df_stock_total_share.loc[(stock_id, trade_date), :] = df_stock_total_share.loc[(stock_id, stock_ipo.ipo_date)]
            except KeyError:
                pass

            try:
                df_stock_free_float_share.loc[(stock_id, trade_date), :] = df_stock_free_float_share.loc[(stock_id, stock_ipo.ipo_date)]
            except KeyError:
                pass

            # df_stock_free_float_share.loc[(stock_id, trade_date), :] = stock_ipo.ipo_amount

            try:
                df_stock_total_market_value.loc[trade_date, stock_id] = \
                    df_stock_total_share.loc[(stock_id, trade_date), 'total_share'] * stock_ipo.ipo_price
            except KeyError:
                pass

        df_stock_total_share = df_stock_total_share.sort_index().reset_index()
        df_stock_total_share.loc[:, 'end_date'] = df_stock_total_share.begin_date.shift(-1)
        df_stock_total_share.loc[df_stock_total_share.stock_id!=df_stock_total_share.stock_id.shift(-1), 'end_date'] = np.nan

        df_stock_free_float_share = df_stock_free_float_share.sort_index().reset_index()
        df_stock_free_float_share.loc[:, 'end_date'] = df_stock_free_float_share.begin_date.shift(-1)
        df_stock_free_float_share.loc[df_stock_free_float_share.stock_id!=df_stock_free_float_share.stock_id.shift(-1), 'end_date'] = np.nan

        return df_stock_total_share, df_stock_free_float_share, df_stock_total_market_value


class StockPortfolio(metaclass=MetaClassPropertyFuncGenerater):

    _ref_dict = {}

    _variable_list_in_data = [
        'index_id',
        'reindex',
        'look_back',
        'trade_dates',
        'trade_dates_total',
        'ser_index_nav',
        'df_index_historical_constituents',
        'stock_pool',
        'stock_pool_total',
        'df_stock_ipo',
        'df_stock_swap',
        'df_stock_status',
        'df_stock_prc',
        'df_stock_ret',
        'df_stock_historical_industry',
        'df_stock_total_share',
        'df_stock_free_float_share',
        'df_stock_total_market_value',
        'stock_market_data',
        'stock_financial_data',
        'df_stock_financial_descriptor',
        'df_dividend',
        'df_pdps_ttm'
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

        if ref in StockPortfolio._ref_dict:
            self._data = StockPortfolio._ref_dict[ref]
        else:
            self._data = StockPortfolio._ref_dict[ref] = StockPortfolioData(index_id, reindex, look_back)

        self._df_stock_pos = None
        self._df_stock_pos_adjusted = None

        self._ser_portfolio_nav = None
        self._ser_portfolio_inc = None
        self._ser_turnover = None

    def calc_portfolio_nav(self, considering_status=False, considering_fee=True):

        self.calc_stock_pos_days()

        # df_stock_pos_adjusted = pd.DataFrame(index=self.trade_dates, columns=self.stock_pool.index)
        # df_stock_pos_adjusted.loc[self.reindex[0]] = self.df_stock_pos.loc[self.reindex[0], :].fillna(0.0)
        stock_pos_adjusted = self.df_stock_pos.loc[self.reindex[0], :].reindex(self.stock_pool_total.index).fillna(0.0)
        arr_stock_pos_adjusted = np.array(stock_pos_adjusted.values.reshape(1, -1))
        ser_portfolio_nav = pd.Series(1.0, index=self.trade_dates, name='nav')
        ser_portfolio_inc = pd.Series(0.0, index=self.trade_dates, name='inc')
        ser_turnover = pd.Series(0.0, index=self.trade_dates, name='turnover')

        for last_trade_date, trade_date in zip(self.trade_dates[:-1], self.trade_dates[1:]):

            stock_pos = stock_pos_adjusted * (self.df_stock_prc.loc[trade_date] / self.df_stock_prc.loc[last_trade_date]).fillna(1.0)

            nav = stock_pos.sum()
            ser_portfolio_inc.loc[trade_date] = nav - 1.0
            ser_portfolio_nav.loc[trade_date] = ser_portfolio_nav.loc[last_trade_date] * nav

            stock_pos.loc[:] = stock_pos / nav

            if trade_date in self.reindex:

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

                if considering_fee:

                    ser_portfolio_nav.loc[trade_date] -= ser_turnover.loc[trade_date] * 0.003
                    ser_portfolio_inc.loc[trade_date] = ser_portfolio_nav.loc[trade_date] / ser_portfolio_nav.loc[last_trade_date] - 1

            else:

                stock_pos_adjusted = copy.deepcopy(stock_pos)
                ser_turnover.loc[trade_date] = 0.0

            if considering_status:
                for _, stock_swap in self.df_stock_swap.loc[self.df_stock_swap.equity_registration_date==trade_date].iterrows():

                    pos = stock_pos_adjusted.loc[stock_swap.transferer_stock_id] \
                        * (stock_swap.conversion_ratio * stock_swap.targetcomp_conversion_prc / stock_swap.transferer_stock_prc)

                    stock_pos_adjusted.loc[stock_swap.targetcomp_stock_id] = pos
                    stock_pos_adjusted.loc[stock_swap.transferer_stock_id] = 0.0

            # df_stock_pos_adjusted.loc[trade_date] = stock_pos_adjusted
            arr_stock_pos_adjusted = np.append(arr_stock_pos_adjusted, stock_pos_adjusted.values.reshape(1, -1), axis=0)

        df_stock_pos_adjusted = pd.DataFrame(arr_stock_pos_adjusted, index=self.trade_dates, columns=self.stock_pool_total.index)

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

        # ser_reindex = pd.Series(self.reindex, index=self.reindex)
        # df_stock_pos = pd.DataFrame(index=self.reindex, columns=self.stock_pool.index)
        # df_stock_pos.loc[:, :] = ser_reindex.apply(self._calc_stock_pos)

        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_count//2)

        res = pool.map(self._calc_stock_pos, self.reindex.to_list())

        pool.close()
        pool.join()

        df_stock_pos = pd.DataFrame(res)

        self._df_stock_pos = df_stock_pos

        return df_stock_pos

    def _calc_stock_pos(self, trade_date):

        raise NotImplementedError('Method \'_calc_stock_pos\' is not defined.')

    def _load_stock_ids(self, trade_date):

        trade_date = trade_date_after(trade_date)

        # stock_pool = caihui_tq_ix_comp.load_index_constituents(self.index_id, date=trade_date.strftime('%Y%m%d'))
        stock_ids = self.df_index_historical_constituents.loc[
            (self.df_index_historical_constituents.in_date<=trade_date) & \
            ((self.df_index_historical_constituents.out_date>=trade_date) | \
            (self.df_index_historical_constituents.out_date.isna()))
        ].set_index('stock_id').index.sort_values()

        return stock_ids

    def _load_stock_price(self, trade_date, stock_ids, fillna=True):

        reindex = self.trade_dates_total[self.trade_dates_total<=trade_date][-self.look_back-1:]

        # df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_prc(stock_ids=stock_ids, reindex=reindex, fill_method='pad')
        df_stock_prc = self.df_stock_prc.reindex(index=reindex, columns=stock_ids)

        if not fillna:

            df_stock_status = self.df_stock_status.loc[reindex, stock_ids]
            df_stock_prc[df_stock_status>2] = np.nan

        return df_stock_prc

    def _load_stock_return(self, trade_date, stock_ids, fillna=True):

        reindex = self.trade_dates_total[self.trade_dates_total<=trade_date][-self.look_back-1:]

        # df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_prc(stock_ids=stock_ids, reindex=reindex, fill_method='pad')
        # df_stock_ret = df_stock_prc.pct_change().iloc[1:]
        df_stock_ret = self.df_stock_ret.reindex(index=reindex[1:], columns=stock_ids)

        if not fillna:

            df_stock_status = self.df_stock_status.loc[reindex[1:], stock_ids]
            df_stock_ret[df_stock_status>2] = np.nan

        return df_stock_ret

    def _load_stock_industry(self, trade_date, stock_ids):

        trade_date = trade_date_after(trade_date)

        df_stock_industry = self.df_stock_historical_industry.loc[
            (self.df_stock_historical_industry.entry_date<=trade_date) & \
            ((self.df_stock_historical_industry.remove_date>=trade_date) | \
            (self.df_stock_historical_industry.remove_date.isna()))
        ].set_index('stock_id').sort_index().loc[stock_ids, ['sw_ind_code', 'sw_lv1_ind_code']]

        return df_stock_industry

    def _calc_portfolio_size(self, size, percentage, lower_bound=1):

        portfolio_size = round(size * percentage)
        portfolio_size = int(max(portfolio_size, lower_bound))

        return portfolio_size

    def portfolio_analysis(self, reindex=None):

        if self.ser_portfolio_nav is None:
            self.calc_portfolio_nav()

        if reindex is None:
            reindex = self.trade_dates

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

        turnover = self.ser_turnover.mean()
        print(f'turnover per day: {turnover}.')

        return portfolio_return, sharpe_ratio

    def portfolio_statistic(self, benchmark_id):

        if self.ser_portfolio_nav is None:
            self.calc_portfolio_nav()

        # ser_benchmark_nav = asset_sp_stock_portfolio_nav.load(benchmark_id) \
            # .nav.rename(benchmark_id)

        ser_benchmark_nav = wind_aindexeodprices.load_a_index_nav_ser(benchmark_id)

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
        ser_stock_volatility.loc[df_stock_ret.isna().sum()>(df_stock_ret.shape[0]//2)] = np.nan

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
        _, ser_market_value_stock_weight = self._calc_market_value(trade_date, momentum_stock_ids)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[momentum_stock_ids] = ser_market_value_stock_weight.fillna(0.0)

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

        ser_stock_free_float_market_value = self._calc_stock_free_float_market_value(trade_date, stock_ids)

        portfolio_size = self._calc_portfolio_size(stock_ids.size, percentage)
        small_size_stock_ids = ser_stock_free_float_market_value.sort_values(ascending=True).iloc[:portfolio_size].index

        # Small size condition quantile as percentage
        ser_small_size_stock_weight = ser_stock_free_float_market_value.loc[small_size_stock_ids]
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

        # df_benchmark_nav = asset_sp_stock_portfolio_nav.load(self.benchmark_id)
        # self._ser_benchmark_nav = df_benchmark_nav.nav.rename(self.benchmark_id)
        # self._ser_benchmark_inc = df_benchmark_nav.inc.rename(self.benchmark_id)
        self._ser_benchmark_nav = self.ser_index_nav
        self._ser_benchmark_inc = self._ser_benchmark_nav.pct_change().iloc[1:]

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

        df_stock_industry = self._load_stock_industry(trade_date, stock_ids)

        ser_industry_free_float_market_value = self._calc_industry_free_float_market_value(trade_date, stock_ids)

        sector_neutral_stock_ids = pd.Index([], name='stock_id')
        ser_sector_neutral_stock_weight = pd.Series(index=stock_ids, name=trade_date)

        ser_percentage_by_industry = self._calc_percentage_by_industry(trade_date, stock_ids, percentage)

        for sw_industry_code in ser_industry_free_float_market_value.index:

            sw_industry_stock_ids = df_stock_industry.loc[df_stock_industry.sw_lv1_ind_code==sw_industry_code].index

            class_name = self.__class__.__name__
            algo = re.sub('StockPortfolioSectorNeutral', '', class_name, count=1)
            algo = re.sub('New', '', algo, count=1)
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
                * ser_industry_free_float_market_value.loc[sw_industry_code]

        sector_neutral_stock_ids = sector_neutral_stock_ids.sort_values()
        ser_sector_neutral_stock_weight = ser_sector_neutral_stock_weight.loc[sector_neutral_stock_ids]
        ser_sector_neutral_stock_weight.loc[:] /= ser_sector_neutral_stock_weight.sum()

        return sector_neutral_stock_ids, ser_sector_neutral_stock_weight

    def _calc_industry_free_float_market_value(self, trade_date, stock_ids, **kwargs):

        ser_stock_free_float_market_value = self._calc_stock_free_float_market_value(trade_date, stock_ids)
        df_stock_industry = self._load_stock_industry(trade_date, stock_ids)

        ser_industry_free_float_market_value = ser_stock_free_float_market_value \
            .rename(index=df_stock_industry.sw_lv1_ind_code).rename_axis('sw_lv1_ind_code').groupby(by='sw_lv1_ind_code').sum()

        return ser_industry_free_float_market_value

    def _calc_percentage_by_industry(self, trade_date, stock_ids, percentage):

        portfolio_size = self._calc_portfolio_size(stock_ids.size, percentage)

        ser_size = self._load_stock_industry(trade_date, stock_ids).groupby(['sw_lv1_ind_code']).size()
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


class StockPortfolioHighDividend(StockPortfolio):

    _kwargs_list = [
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioHighDividend, self).__init__(index_id, reindex, look_back, **kwargs)

    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        high_dividend_stock_ids, ser_high_dividend_stock_weight = self._calc_high_dividend(trade_date, stock_ids, self.percentage)

        # low_volatility_stock_id, low_volatility_volatility = self._calc_low_volatility(trade_date, ser_pdps_ttm_selected.index, self.low_volatility_percentage)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[high_dividend_stock_ids] = ser_high_dividend_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_high_dividend(self, trade_date, stock_ids, percentage):

        df_stock_dividend_recent3years = self._load_stock_dividend_recent3years(trade_date, stock_ids)

        ser_stock_dividend_num_recent3years = df_stock_dividend_recent3years.stock_id.groupby(df_stock_dividend_recent3years.stock_id).count()
        ser_stock_dividend_num_recent3years.loc[ser_stock_dividend_num_recent3years>0] = 0
        ser_stock_dividend_num_recent3years = ser_stock_dividend_num_recent3years.reindex(stock_ids).fillna(0)

        ser_pdps_ttm = self._load_stock_pdps_ttm(trade_date, stock_ids)

        df = pd.DataFrame({'dividend_num':ser_stock_dividend_num_recent3years, 'pdps_ttm': ser_pdps_ttm})

        portfolio_size = self._calc_portfolio_size(stock_ids.size, percentage)
        high_dividend_stock_ids = df.sort_values(by=['dividend_num', 'pdps_ttm'], ascending=False).iloc[:portfolio_size].index

        ser_high_dividend_stock_weight = ser_pdps_ttm.loc[high_dividend_stock_ids]
        ser_high_dividend_stock_weight /= ser_high_dividend_stock_weight.sum()

        return high_dividend_stock_ids, ser_high_dividend_stock_weight

    def _load_stock_pdps_ttm(self, trade_date, stock_ids):

        df_pdps_ttm = self.df_pdps_ttm.loc[trade_date+relativedelta(years=-3):trade_date, stock_ids]
        df_pdps_ttm = 1.0 / df_pdps_ttm

        ser_pdps_ttm_recent3years_avg = df_pdps_ttm.mean(axis=0)

        return ser_pdps_ttm_recent3years_avg

    def _load_stock_dividend_recent3years(self, trade_date, stock_ids):

        df_stock_dividend_recent3years = self.df_dividend.loc[
            (self.df_dividend.ann_date<=trade_date) &
            (self.df_dividend.ann_date>=trade_date+relativedelta(years=-3, month=1, day=1)) &
            (self.df_dividend.stock_id.isin(stock_ids)) &
            (self.df_dividend.cash_dvd_per_sh_pre_tax>0.0)
        ]

        return df_stock_dividend_recent3years


class StockPortfolioHighDividendLowVolatility(StockPortfolioHighDividend, StockPortfolioLowVolatility):

    _kwargs_list = [
        'percentage_high_dividend',
        'percentage_low_volatility'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioHighDividendLowVolatility, self).__init__(index_id, reindex, look_back, **kwargs)

    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        high_dividend_stock_ids, ser_high_dividend_stock_weight = self._calc_high_dividend(trade_date, stock_ids, self.percentage_high_dividend)
        high_dividend_low_volatility_stock_ids, ser_low_volatility_stock_weight = self._calc_low_volatility(trade_date, high_dividend_stock_ids, self.percentage_low_volatility)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[high_dividend_low_volatility_stock_ids] = ser_low_volatility_stock_weight.fillna(0.0)

        return stock_pos

    def _calc_high_dividend(self, trade_date, stock_ids, percentage):

        df_stock_dividend_recent3years = self._load_stock_dividend_recent3years(trade_date, stock_ids)

        ser_stock_dividend_num_recent3years = df_stock_dividend_recent3years.stock_id.groupby(df_stock_dividend_recent3years.stock_id).count()
        ser_stock_dividend_num_recent3years.loc[ser_stock_dividend_num_recent3years>3] = 3
        ser_stock_dividend_num_recent3years = ser_stock_dividend_num_recent3years.reindex(stock_ids).fillna(0)

        ser_pdps_ttm = self._load_stock_pdps_ttm(trade_date, stock_ids)

        df = pd.DataFrame({'dividend_num':ser_stock_dividend_num_recent3years, 'pdps_ttm': ser_pdps_ttm})

        portfolio_size = self._calc_portfolio_size(stock_ids.size, percentage)
        high_dividend_stock_ids = df.sort_values(by=['dividend_num', 'pdps_ttm'], ascending=False).iloc[:portfolio_size].index

        ser_high_dividend_stock_weight = ser_pdps_ttm.loc[high_dividend_stock_ids]
        ser_high_dividend_stock_weight /= ser_high_dividend_stock_weight.sum()

        return high_dividend_stock_ids, ser_high_dividend_stock_weight

    def _load_stock_pdps_ttm(self, trade_date, stock_ids):

        df_pdps_ttm = self.df_pdps_ttm.loc[trade_date+relativedelta(years=-3):trade_date, stock_ids]

        ser_pdps_ttm_recent3years_avg = df_pdps_ttm.mean(axis=0)

        return ser_pdps_ttm_recent3years_avg

    def _load_stock_dividend_recent3years(self, trade_date, stock_ids):

        df_stock_dividend_recent3years = self.df_dividend.loc[
            (self.df_dividend.ann_date<=trade_date) &
            (self.df_dividend.ann_date>=trade_date+relativedelta(years=-3, month=1, day=1)) &
            (self.df_dividend.stock_id.isin(stock_ids)) &
            (self.df_dividend.cash_dvd_per_sh_pre_tax>0.0)
        ]

        return df_stock_dividend_recent3years


class StockPortfolioLowVolatilityNew(StockPortfolioLowVolatility):

    _kwargs_list = [
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioLowVolatilityNew, self).__init__(index_id, reindex, look_back, **kwargs)

    # Reference: http://www.csindex.com.cn/zh-CN/indices/index-detail/000803
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        last_adjustment_trade_date = last_adjustment_date(trade_date, self.index_id)
        last_adjusted_stock_ids = self._load_stock_ids(last_adjustment_trade_date)

        low_volatility_stock_ids, _ = self._calc_low_volatility(last_adjustment_trade_date, last_adjusted_stock_ids, self.percentage)
        low_volatility_stock_ids = low_volatility_stock_ids.intersection(stock_ids)
        _, ser_low_volatility_stock_weight = self._calc_low_volatility(trade_date, low_volatility_stock_ids, 1.0)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[low_volatility_stock_ids] = ser_low_volatility_stock_weight.fillna(0.0)

        return stock_pos


class StockPortfolioLowBetaLowVolatilityNew(StockPortfolioLowBetaLowVolatility):

    _kwargs_list = [
        'percentage_low_beta',
        'percentage_low_volatility',
        'benchmark_id'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioLowBetaLowVolatilityNew, self).__init__(index_id, reindex, look_back, **kwargs)

    # Reference: http://www.csindex.com.cn/zh-CN/indices/index-detail/930985
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        last_adjustment_trade_date = last_adjustment_date(trade_date, self.index_id)
        last_adjusted_stock_ids = self._load_stock_ids(last_adjustment_trade_date)

        low_beta_stock_ids, _ = self._calc_low_beta(last_adjustment_trade_date, last_adjusted_stock_ids, self.percentage_low_beta)
        low_beta_stock_ids = low_beta_stock_ids.intersection(stock_ids)
        low_beta_low_volatility_stock_ids, _ = self._calc_low_volatility(trade_date, low_beta_stock_ids, self.percentage_low_volatility)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[low_beta_low_volatility_stock_ids] = 1.0 / low_beta_low_volatility_stock_ids.size

        return stock_pos


class StockPortfolioSectorNeutralNew(StockPortfolioSectorNeutral):

    _kwargs_list = [
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioSectorNeutralNew, self).__init__(index_id, reindex, look_back, **kwargs)

    # Reference: http://www.csindex.com.cn/zh-CN/indices/index-detail/930846
    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        last_adjustment_trade_date = last_adjustment_date(trade_date, self.index_id)
        last_adjusted_stock_ids = self._load_stock_ids(last_adjustment_trade_date)

        sector_neutral_stock_ids, _ = self._calc_sector_neutral(last_adjustment_trade_date, last_adjusted_stock_ids, self.percentage)
        sector_neutral_stock_ids = sector_neutral_stock_ids.intersection(stock_ids)
        _, ser_sector_neutral_stock_weight = self._calc_sector_neutral(trade_date, sector_neutral_stock_ids, 1.0)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[sector_neutral_stock_ids] = ser_sector_neutral_stock_weight.fillna(0.0)

        return stock_pos


class StockPortfolioSectorNeutralLowVolatilityNew(StockPortfolioSectorNeutralNew, StockPortfolioLowVolatility):

    _kwargs_list = [
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioSectorNeutralLowVolatilityNew, self).__init__(index_id, reindex, look_back, **kwargs)


class StockPortfolioHighDividendNew(StockPortfolioHighDividend):

    _kwargs_list = [
        'percentage'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioHighDividendNew, self).__init__(index_id, reindex, look_back, **kwargs)

    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        last_adjustment_trade_date = last_adjustment_date(trade_date, self.index_id)
        last_adjusted_stock_ids = self._load_stock_ids(last_adjustment_trade_date)

        high_dividend_stock_ids, _ = self._calc_high_dividend(last_adjustment_trade_date, last_adjusted_stock_ids, self.percentage)
        high_dividend_stock_ids = high_dividend_stock_ids.intersection(stock_ids)
        _, ser_high_dividend_stock_weight = self._calc_high_dividend(trade_date, high_dividend_stock_ids, 1.0)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[high_dividend_stock_ids] = ser_high_dividend_stock_weight.fillna(0.0)

        return stock_pos


class StockPortfolioHighDividendLowVolatilityNew(StockPortfolioHighDividendLowVolatility):

    _kwargs_list = [
        'percentage_high_dividend',
        'percentage_low_volatility'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioHighDividendLowVolatilityNew, self).__init__(index_id, reindex, look_back, **kwargs)

    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        last_adjustment_trade_date = last_adjustment_date(trade_date, self.index_id)
        last_adjusted_stock_ids = self._load_stock_ids(last_adjustment_trade_date)

        high_dividend_stock_ids, ser_high_dividend_stock_weight = self._calc_high_dividend(last_adjustment_trade_date, last_adjusted_stock_ids, self.percentage_high_dividend)
        high_dividend_stock_ids = high_dividend_stock_ids.intersection(stock_ids)
        high_dividend_low_volatility_stock_ids, ser_low_volatility_stock_weight = self._calc_low_volatility(trade_date, high_dividend_stock_ids, self.percentage_low_volatility)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[high_dividend_low_volatility_stock_ids] = ser_low_volatility_stock_weight.fillna(0.0)

        return stock_pos


class StockPortfolioIndustry(StockPortfolio):

    _kwargs_list = [
        'sw_industry_code'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioIndustry, self).__init__(index_id, reindex, look_back, **kwargs)

    def _calc_stock_pos(self, trade_date):

        sw_industry_stock_ids = self._load_stock_ids_by_industry(trade_date)

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

    def _load_stock_ids_by_industry(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)
        df_stock_industry = self._load_stock_industry(trade_date, stock_ids)
        stock_ids_by_industry = df_stock_industry.loc[df_stock_industry.sw_lv1_ind_code==self.sw_industry_code].index

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


class StockPortfolioHRP(StockPortfolio):

    def __init__(self, index_id, reindex, look_back, **kwargs):

        super(StockPortfolioHRP, self).__init__(index_id, reindex, look_back, **kwargs)

    def _calc_stock_pos(self, trade_date):

        stock_ids = self._load_stock_ids(trade_date)

        HRP_stock_ids, ser_HRP_weight = self._calc_HRP(trade_date, stock_ids, 1.0)

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[HRP_stock_ids] = ser_HRP_weight.fillna(0.0)

        return stock_pos

    def _calc_HRP(self, trade_date, stock_ids, percentage, **kwargs):

        df_stock_ret = self._load_stock_return(trade_date, stock_ids, fillna=False)
        df_stock_ret = df_stock_ret.loc[:, df_stock_ret.isna().sum() < (df_stock_ret.shape[0] // 3)]
        HRP_stock_ids = df_stock_ret.columns
        ser_HRP_weight = self.HRP(df_stock_ret)

        return HRP_stock_ids, ser_HRP_weight

    def _calc_ivp(self, cov):

        # Compute the inverse-variance percentage

        ivp = 1.0 / np.diag(cov)
        ivp /= ivp.sum()

        return ivp

    def _calc_cluster_var(self, cov, cItems):

        # Compute variance per cluster

        cov = cov.loc[cItems, cItems]
        w = self._calc_ivp(cov).reshape(-1, 1)
        cVar = np.dot(np.dot(w.T, cov), w)[0, 0]

        return cVar

    def HRP(self, Underlying_Ret):

        '''
        The input dataframe should be the daily return data with timstamp as index of the portfolio, where are needed to be compared

        Hierarchical Risk Parity Portfolio: A fucking innovative methodologies
        Questions here: Stabilization of Graph theory is trade-off between the model risk e.g. the norm or numerical distance selection?
        Ref: Marcos López de Prado 2015
        '''

        if not isinstance(Underlying_Ret, pd.DataFrame):
            raise ValueError('Stock Return matrix is not a DataFrame')

        Underlying_Ret_corr = Underlying_Ret.corr()
        Underlying_Ret_cov = Underlying_Ret.cov()

        Corr_Distance = np.sqrt((1.0 - Underlying_Ret_corr) / 2.0)
        Corr_Distance_Linker = hierarchy.linkage(Corr_Distance, 'single')

        leaves_list = hierarchy.leaves_list(Corr_Distance_Linker)
        sorted_index = Underlying_Ret_cov.index[leaves_list]

        Underlying_Alloocated_Weight = pd.Series(1, index=sorted_index)
        # initialize all items in one cluster
        cItems = [sorted_index.to_list()]

        while len(cItems) > 0:
            # bi-section
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(cItems), 2):
                # cluster 1
                cItems0 = cItems[i]
                # cluster 2
                cItems1 = cItems[i + 1]
                cVar0 = self._calc_cluster_var(Underlying_Ret_cov, cItems0)
                cVar1 = self._calc_cluster_var(Underlying_Ret_cov, cItems1)
                alpha = 1 - cVar0 / (cVar0 + cVar1)
                # weight 1
                Underlying_Alloocated_Weight[cItems0] *= alpha
                # weight 2
                Underlying_Alloocated_Weight[cItems1] *= 1 - alpha

        return Underlying_Alloocated_Weight.sort_index()


class StockPortfolioMultiFactor(StockPortfolio):

    _kwargs_list = [
        'factor_weighted_method',
        'select_num',
        'style_factor',
        'trading_frequency'
    ]

    def __init__(self, index_id, reindex, look_back, **kwargs):
        super(StockPortfolioMultiFactor, self).__init__(index_id, reindex, look_back, **kwargs)
        self.df_stock_factor_exposure = factor_sf_stock_factor_exposure.load_a_stock_factor_exposure(stock_ids=self.stock_pool.index, style_factor=self.style_factor)
        self.df_factor_return = factor_sf_stock_factor_exposure.load_factor_return(self.style_factor)
        self.df_factor_residual_return = self._calc_residual_return()

    def _calc_stock_pos(self, trade_date):
        # calculate factor weight
        min_count = 50
        change_pos_dates_t = self.reindex[::self.trading_frequency].copy()
        temp_variable = 0
        if trade_date not in change_pos_dates_t:
            change_pos_dates_t = change_pos_dates_t[change_pos_dates_t < trade_date]
            trade_date = change_pos_dates_t[-1]
        else:
            temp_variable = 1
        last_trade_date = self.reindex_total[list(self.reindex_total).index(trade_date) - 1]
        df_factor_residual_return_t = self.df_factor_residual_return.loc[:last_trade_date].iloc[::-self.trading_frequency].sort_index().copy()
        if df_factor_residual_return_t.shape[0] < min_count:
            ser_weight_t = pd.Series(data=1/len(self.style_factor), index=self.style_factor)
            negative_factor = ['quality_risk', 'linear_size', 'non_linear_size', 'liquidity', 'short_term_reverse']
            for i_index in ser_weight_t.index:
                if i_index in negative_factor:
                    ser_weight_t[i_index] = -1 * ser_weight_t[i_index]
        else:
            if self.factor_weighted_method == 'sharpe_ratio':
                numerator_t = self.df_factor_return.loc[:trade_date].iloc[:-1].mean()
                denominator_t = df_factor_residual_return_t.std()
                ser_weight_t = numerator_t / denominator_t
            elif self.factor_weighted_method == 'min_variance':
                negative_factor = ['quality_risk', 'linear_size', 'non_linear_size', 'liquidity', 'short_term_reverse']
                for i_index in df_factor_residual_return_t.columns:
                    if i_index in negative_factor:
                        df_factor_residual_return_t[i_index] = -1 * df_factor_residual_return_t[i_index]
                covariance_t = calc_covariance.calc_covariance(data=df_factor_residual_return_t, lookback_period=128, H_L_vol=32, Lags_vol=2, H_L_corr=64, Lags_corr=2, Predict_period=1)
                P_t = matrix(covariance_t)
                q_t = matrix(np.zeros((covariance_t.shape[0], 1)))
                G_t = matrix(np.eye(covariance_t.shape[0]) * -1)
                h_t = matrix(np.zeros(covariance_t.shape[0]))
                A_t = matrix(np.ones(covariance_t.shape[0]).reshape(1, -1))
                b_t = matrix([1.0])
                solvers.options['show_progress'] = False  # Notice
                sol = solvers.qp(P=P_t, q=q_t, G=G_t, h=h_t, A=A_t, b=b_t)
                opt_t = np.array(sol['x']).T[0]
                ser_weight_t = pd.Series(opt_t, index=df_factor_residual_return_t.columns)
                #
                negative_factor = ['quality_risk', 'linear_size', 'non_linear_size', 'liquidity', 'short_term_reverse']
                for i_index in ser_weight_t.index:
                    if i_index in negative_factor:
                        ser_weight_t[i_index] = -1 * ser_weight_t[i_index]
            ser_weight_t = ser_weight_t / ser_weight_t.abs().sum()
        # calculate stock score
        if temp_variable == 1:
            print(ser_weight_t)
        stock_ids = self._load_stock_ids(trade_date)
        df_stock_factor_exposure_t = self.df_stock_factor_exposure.loc[last_trade_date].reindex(stock_ids).copy()
        df_stock_factor_exposure_t['score'] = 0.0
        for i_style in self.style_factor:
            df_stock_factor_exposure_t['score'] = df_stock_factor_exposure_t['score'] + df_stock_factor_exposure_t[i_style] * ser_weight_t[i_style]
        df_stock_factor_exposure_t.sort_values(by='score', inplace=True, ascending=False)
        stock_pos = pd.Series(1.0 / self.select_num, index=df_stock_factor_exposure_t.iloc[:self.select_num].index, name=trade_date)
        return stock_pos

    def _calc_residual_return(self):
        trade_dates = self.df_factor_return.sort_index().index
        df_factor_residual_return = pd.DataFrame()
        for last_trade_date, trade_date in zip(trade_dates[:-self.trading_frequency], trade_dates[self.trading_frequency:]):
            expected_return_t = self.df_factor_return.loc[:last_trade_date].mean(axis=0)
            real_return_t = self.df_factor_return.loc[trade_date]
            df_factor_residual_return_t = pd.DataFrame(real_return_t - expected_return_t).T
            df_factor_residual_return_t['trade_date'] = trade_date
            df_factor_residual_return = df_factor_residual_return.append(df_factor_residual_return_t, ignore_index=True)
        return df_factor_residual_return.set_index('trade_date')


class FactorPortfolioData(StockPortfolioData):

    def __init__(self, stock_portfolio_ids, reindex, look_back):

        self.stock_portfolio_ids = stock_portfolio_ids

        df_stock_portfolio_info = asset_sp_stock_portfolio.load_by_id(
            portfolio_ids=self.stock_portfolio_ids
        )
        if not (df_stock_portfolio_info.sp_type==1).all():
            raise ValueError('Types of Stock Portfolios should be 1.')

        self.reindex = reindex
        self.look_back = look_back

        self.trade_dates = ATradeDate.trade_date(
            begin_date=self.reindex[0],
            end_date=self.reindex[-1]
        ).rename('trade_date')

        self.trade_dates_total = ATradeDate.trade_date(
            begin_date=self.reindex[0],
            end_date=self.reindex[-1],
            lookback=self.look_back+1
        ).rename('trade_date')

        self.df_factor_prc = asset_sp_stock_portfolio_nav.load_portfolio_nav(
            portfolio_ids=stock_portfolio_ids,
            reindex=self.trade_dates_total
        )
        self.df_factor_ret = self.df_factor_prc.pct_change().iloc[1:]

        self.df_stock_portfolio_pos = asset_sp_stock_portfolio_pos.load_portfolio_pos(
            portfolio_ids=stock_portfolio_ids,
            reindex=self.trade_dates
        ).swaplevel(0, 1).pos.unstack()

        self.stock_pool = wind_asharedescription.load_a_stock_code_info(
            stock_ids=self.df_stock_portfolio_pos.columns
        )

        self.stock_pool_total = self.stock_pool.copy(deep=True)

        self.df_stock_swap = wind_asharestockswap.load_a_stock_swap(
            transferer_stock_ids=self.stock_pool.index
        )
        self.df_stock_swap.dropna(inplace=True)

        self.df_stock_ipo = wind_ashareipo.load_a_stock_ipo_info(
            stock_ids=self.stock_pool.index
        )

        self.df_stock_status = wind_ashareeodprices.load_a_stock_status(
            stock_ids=self.stock_pool_total.index,
            reindex=self.trade_dates_total
        )

        self.df_stock_prc, self.df_stock_ret = self._load_stock_price_and_return()


class FactorPortfolio(StockPortfolio, metaclass=MetaClassPropertyFuncGenerater):

    _ref_dict = {}

    _variable_list_in_data = [
        'stock_portfolio_ids',
        'reindex',
        'look_back',
        'trade_dates',
        'trade_dates_total',
        'df_factor_prc',
        'df_factor_ret',
        'df_stock_portfolio_pos',
        'stock_pool',
        'stock_pool_total',
        'df_stock_ipo',
        'df_stock_swap',
        'df_stock_status',
        'df_stock_prc',
        'df_stock_ret'
    ]

    _instance_variable_list = [
        'df_factor_pos',
        'df_stock_pos',
        'df_stock_pos_adjusted',
        'ser_portfolio_nav',
        'ser_portfolio_inc',
        'ser_turnover'
    ]

    _kwargs_list = []

    def __init__(self, stock_portfolio_ids, reindex, look_back, **kwargs):

        for key in self._kwargs_list:

            if key not in kwargs:
                raise TypeError(f'__init__() missing 1 required positional argument: \'{key}\'')
            value = kwargs.get(key)
            setattr(self, f'_{key}', value)
            setattr(self.__class__, key, property(MetaClassPropertyFuncGenerater.generate_func_for_instance_variable(key)))

        ref = f'{stock_portfolio_ids}, {reindex}, {look_back}'
        sha1 = hashlib.sha1()
        sha1.update(ref.encode('utf-8'))
        ref = sha1.hexdigest()

        if ref in FactorPortfolio._ref_dict:
            self._data = FactorPortfolio._ref_dict[ref]
        else:
            self._data = FactorPortfolio._ref_dict[ref] = FactorPortfolioData(stock_portfolio_ids, reindex, look_back)

        self._df_factor_pos = None
        self._df_stock_pos = None
        self._df_stock_pos_adjusted = None

        self._ser_portfolio_nav = None
        self._ser_portfolio_inc = None
        self._ser_turnover = None

    def calc_portfolio_nav(self, considering_status=True, considering_fee=False):

        self.calc_factor_pos_days()
        ser_portfolio_nav = super(FactorPortfolio, self).calc_portfolio_nav(considering_status, considering_fee)

        return ser_portfolio_nav

    def _calc_stock_pos(self, trade_date):

        # Reference: http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dot.html
        # self @ other <=> self.dot(other)
        stock_pos = self.df_factor_pos.loc[trade_date].fillna(0.0) @ self.df_stock_portfolio_pos.loc[trade_date].fillna(0.0)
        stock_pos /= stock_pos.sum()
        # stock_pos = np.dot(self.df_factor_pos.loc[trade_date], self.df_stock_portfolio_pos.loc[trade_date])

        return stock_pos

    def calc_factor_pos_days(self):

        if self.df_factor_pos is not None:
            return self.df_factor_pos

        ser_reindex = pd.Series(self.reindex, index=self.reindex)
        df_factor_pos = pd.DataFrame(index=self.reindex, columns=self.stock_portfolio_ids)
        df_factor_pos.loc[:, :] = ser_reindex.apply(self._calc_factor_pos)

        self._df_factor_pos = df_factor_pos

        return df_factor_pos

    def _calc_factor_pos(self, trade_date):

        raise NotImplementedError('Method \'_calc_factor_pos\' is not defined.')

    def _load_factor_return(self, trade_date, factor_ids, fillna=True):

        reindex = self.trade_dates_total[self.trade_dates_total<=trade_date][-self.look_back-1:]

        df_factor_ret = self.df_factor_ret.reindex(index=reindex[1:], columns=factor_ids)

        return df_factor_ret


class FactorPortfolioEqualWeight(FactorPortfolio):

    _kwargs_list = []

    def __init__(self, stock_portfolio_ids, reindex, look_back, **kwargs):

        super(FactorPortfolioEqualWeight, self).__init__(stock_portfolio_ids, reindex, look_back, **kwargs)

    def _calc_factor_pos(self, trade_date):

        factor_pos = pd.Series(1.0/self.stock_portfolio_ids.size, index=self.stock_portfolio_ids, name=trade_date)

        return factor_pos


class FactorPortfolioMaxDecorrelation(FactorPortfolio):

    _kwargs_list = []

    def __init__(self, stock_portfolio_ids, reindex, look_back, **kwargs):

        super(FactorPortfolioMaxDecorrelation, self).__init__(stock_portfolio_ids, reindex, look_back, **kwargs)

    def _calc_factor_pos(self, trade_date):

        max_decorrelation_weight = self._calc_max_decorrelation(trade_date, self.stock_portfolio_ids)

        factor_pos = max_decorrelation_weight.reindex(self.stock_portfolio_ids).fillna(0.0).rename(trade_date)

        return factor_pos

    def _calc_max_decorrelation(self, trade_date, factor_ids):

        df_factor_ret = self._load_factor_return(trade_date, factor_ids, fillna=False)
        corr_mat = df_factor_ret.corr()
        max_decorrelation_weight = FactorPortfolioMaxDecorrelation.max_decorrelation_portfolio(corr_mat)

        return max_decorrelation_weight

    @staticmethod
    def max_decorrelation_portfolio(corr_mat, allow_short=False):

        '''
        Computes the maximum decorrelation portfolio.

        Note: As the variance is not invariant with respect
        to leverage, it is not possible to construct non-trivial
        market neutral minimum variance portfolios. This is because
        the variance approaches zero with decreasing leverage,
        i.e. the market neutral portfolio with minimum variance
        is not invested at all.

        Parameters
        ----------
        corr_mat: pandas.DataFrame
            Correlation matrix of asset returns.
        allow_short: bool, optional
            If 'False' construct a long-only portfolio.
            If 'True' allow shorting, i.e. negative weights.

        Returns
        -------
        weights: pandas.Series
            Optimal asset weights.
        '''

        if not isinstance(corr_mat, pd.DataFrame):
            raise TypeError("Covariance matrix is not a DataFrame")

        n = corr_mat.shape[0]

        P = cvxopt.matrix(corr_mat.values)
        q = cvxopt.matrix(0.0, (n, 1))

        # Constraints Gx <= h
        if not allow_short:
            # x >= 0
            G = cvxopt.matrix(-np.identity(n))
            h = cvxopt.matrix(0.0, (n, 1))
        else:
            G = None
            h = None

        # Constraints Ax = b
        # sum(x) = 1
        A = cvxopt.matrix(1.0, (1, n))
        b = cvxopt.matrix(1.0)

        # Solve
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        if sol['status'] != 'optimal':
            warnings.warn("Convergence problem")

        # Put weights into a labeled series
        weights = pd.Series(sol['x'], index=corr_mat.index)

        return weights


# ValueError: domain error
class FactorPortfolioTangency(FactorPortfolio):

    _kwargs_list = []

    def __init__(self, stock_portfolio_ids, reindex, look_back, **kwargs):

        super(FactorPortfolioTangency, self).__init__(stock_portfolio_ids, reindex, look_back, **kwargs)

    def _calc_factor_pos(self, trade_date):

        tangency_weight = self._calc_tangency(trade_date, self.stock_portfolio_ids)

        factor_pos = tangency_weight.reindex(self.stock_portfolio_ids).fillna(0.0).rename(trade_date)

        return factor_pos

    def _calc_tangency(self, trade_date, factor_ids):

        df_factor_ret = self._load_factor_return(trade_date, factor_ids, fillna=False)
        cov_mat = df_factor_ret.cov()
        exp_rets = df_factor_ret.mean()
        tangency_weight = FactorPortfolioTangency.tangency_portfolio(cov_mat, exp_rets)

        return tangency_weight

    @staticmethod
    def tangency_portfolio(cov_mat, exp_rets, allow_short=False):

        '''
        Computes a tangency portfolio, i.e. a maximum Sharpe ratio portfolio.

        Note: As the Sharpe ratio is not invariant with respect
        to leverage, it is not possible to construct non-trivial
        market neutral tangency portfolios. This is because for
        a positive initial Sharpe ratio the sharpe grows unbound
        with increasing leverage.

        Parameters
        ----------
        cov_mat: pandas.DataFrame
            Covariance matrix of asset returns.
        exp_rets: pandas.Series
            Expected asset returns (often historical returns).
        allow_short: bool, optional
            If 'False' construct a long-only portfolio.
            If 'True' allow shorting, i.e. negative weights.

        Returns
        -------
        weights: pandas.Series
            Optimal asset weights.
        '''

        if not isinstance(cov_mat, pd.DataFrame):
            raise TypeError("Covariance matrix is not a DataFrame")

        if not isinstance(exp_rets, pd.Series):
            raise TypeError("Expected returns is not a Series")

        if not cov_mat.index.equals(exp_rets.index):
            raise ValueError("Indices do not match")

        n = cov_mat.shape[0]

        P = cvxopt.matrix(cov_mat.values)
        q = cvxopt.matrix(0.0, (n, 1))

        # Constraints Gx <= h
        if not allow_short:
            # exp_rets*x >= 1 and x >= 0
            G = cvxopt.matrix(np.vstack((-exp_rets.values, -np.identity(n))))
            h = cvxopt.matrix(np.vstack((-1.0, np.zeros((n, 1)))))
        else:
            # exp_rets*x >= 1
            G = cvxopt.matrix(-exp_rets.values).T
            h = cvxopt.matrix(-1.0)

        # Constraints Ax = b
        # sum(x) = 1
        A = cvxopt.matrix(1.0, (1, n))
        b = cvxopt.matrix(1.0)

        # Solve
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        if sol['status'] != 'optimal':
            warnings.warn("Convergence problem")

        # Put weights into a labeled series
        weights = pd.Series(sol['x'], index=cov_mat.index)

        # Rescale weights, so that sum(weights) = 1
        weights /= weights.sum()

        return weights


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
    stock_portfolio_ids = pd.Index(['SP.000000', 'SP.000100', 'SP.000200', 'SP.000400', 'SP.000800', 'SP.000900'])

    begin_date = '2013-03-04'
    end_date = '2019-03-29'
    look_back = 244
    reindex = ATradeDate.trade_date(begin_date=begin_date, end_date=end_date).rename('trade_date')

    dict_portfolio = {}
    df_portfolio_nav = pd.DataFrame()
    style_factor=['value', 'dividend_yield', 'quality_earnings', 'quality_risk', 'growth', 'linear_size', 'beta', 'liquidity']
    dict_portfolio['MultiFactor'] = StockPortfolioMultiFactor(index_id, trade_dates, look_back, factor_weighted_method='sharpe_ratio', select_num=100, style_factor=style_factor, trading_frequency=10)
    df_portfolio_nav['MultiFactor'] = dict_portfolio['MultiFactor'].calc_portfolio_nav()
    set_trace()

    # dict_portfolio['MarketCap'] = StockPortfolioMarketValue(index_id, reindex, look_back)
    # df_portfolio_nav['MarketCap'] = dict_portfolio['MarketCap'].calc_portfolio_nav()

    # dict_portfolio['EqualWeight'] = StockPortfolioEqualWeight(index_id, reindex, look_back)
    # df_portfolio_nav['EqualWeight'] = dict_portfolio['EqualWeight'].calc_portfolio_nav()

    # dict_portfolio['LowVolatility'] = StockPortfolioLowVolatility(index_id, reindex, look_back, percentage=0.3)
    # df_portfolio_nav['LowVolatility'] = dict_portfolio['LowVolatility'].calc_portfolio_nav()
    # dict_portfolio['LowVolatility'].portfolio_analysis()
    # dict_portfolio['LowVolatility'].portfolio_statistic('CS.000906')

    # dict_portfolio['Momentum'] = StockPortfolioMomentum(index_id, reindex, look_back, percentage=0.3, exclusion=20)
    # tdf_portfolio_nav['Momentum'] = dict_portfolio['Momentum'].calc_portfolio_nav()

    # dict_portfolio['SmallSize'] = StockPortfolioSmallSize(index_id, reindex, look_back, percentage=0.3)
    # df_portfolio_nav['SmallSize'] = dict_portfolio['SmallSize'].calc_portfolio_nav()

    # dict_portfolio['LowBeta'] = StockPortfolioLowBeta(index_id, reindex, look_back, percentage=0.3, benchmark_id='CS.000906')
    # df_portfolio_nav['LowBeta'] = dict_portfolio['LowBeta'].calc_portfolio_nav()

    # dict_portfolio['HighBeta'] = StockPortfolioHighBeta(index_id, reindex, look_back, percentage=0.3, benchmark_id='CS.000906')
    # df_portfolio_nav['HighBeta'] = dict_portfolio['HighBeta'].calc_portfolio_nav()

    # dict_portfolio['HighBetaAndLowBeta'] = StockPortfolioHighBetaAndLowBeta(index_id, reindex, look_back, percentage=0.6, benchmark_id='CS.000906', factor_weight={'high_beta': 0.5, 'low_beta':0.5})
    # df_portfolio_nav['HighBetaAndLowBeta'] = dict_portfolio['HighBetaAndLowBeta'].calc_portfolio_nav()

    # dict_portfolio['LowBetaLowVolatility'] = StockPortfolioLowBetaLowVolatility(index_id, reindex, look_back, percentage_low_beta=0.6, percentage_low_volatility=0.5, benchmark_id='CS.000906')
    # df_portfolio_nav['LowBetaLowVolatility'] = dict_portfolio['LowBetaLowVolatility'].calc_portfolio_nav()

    # dict_portfolio['SectorNeutralLowVolatility'] = StockPortfolioSectorNeutralLowVolatility(index_id, reindex, look_back)
    # df_portfolio_nav['SectorNeutralLowVolatility'] = dict_portfolio['SectorNeutralLowVolatility'].calc_portfolio_nav()

    # dict_portfolio['SectorNeutralLowBeta'] = StockPortfolioSectorNeutralLowBeta(index_id, reindex, look_back, benchmark_id='CS.000905')
    # df_portfolio_nav['SectorNeutralLowBeta'] = dict_portfolio['SectorNeutralLowBeta'].calc_portfolio_nav()

    # dict_portfolio['FamaMacbethRegression'] = StockPortfolioFamaMacbethRegression(index_id, reindex, look_back, trading_frequency=10, percentage=0.15)
    # df_portfolio_nav['FamaMacbethRegression'] = dict_portfolio['FamaMacbethRegression'].calc_portfolio_nav()

    # dict_portfolio['FactorEqualWeight'] = FactorPortfolioEqualWeight(stock_portfolio_ids, reindex, look_back)
    # df_portfolio_nav['FactorEqualWeight'] = dict_portfolio['FactorEqualWeight'].calc_portfolio_nav()

    # dict_portfolio['FactorMaxDecorrelation'] = FactorPortfolioMaxDecorrelation(stock_portfolio_ids, reindex, look_back)
    # df_portfolio_nav['FactorMaxDecorrelation'] = dict_portfolio['FactorMaxDecorrelation'].calc_portfolio_nav()

    # dict_portfolio['FactorTangency']= FactorPortfolioTangency(stock_portfolio_ids, reindex, look_back)
    # df_portfolio_nav['FactorTangency'] = dict_portfolio['FactorTangency'].calc_portfolio_nav()

    # df_portfolio_nav.to_csv('df_portfolio_nav.csv')
    # set_trace()

    # multiprocessing_calc_portfolio_nav_by_industry('IndustryLowVolatility', index_id, reindex, look_back, percentage=0.30)

    # multiprocessing_calc_portfolio_nav_by_industry('IndustryMomentum', index_id, reindex, look_back, percentage=0.3, exclusion=30)
    # set_trace()

