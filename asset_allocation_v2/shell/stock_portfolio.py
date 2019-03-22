#coding=utf-8
'''
Created on: Mar. 11, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import sys
import logging
import warnings
import click
from sqlalchemy import MetaData, Table, select, func, and_
import multiprocessing
import numpy as np
import pandas as pd
import math
import hashlib
import copy
# from ipdb import set_trace
sys.path.append('shell')
from db import database
from db import caihui_tq_ix_comp, caihui_tq_qt_index, caihui_tq_sk_basicinfo, caihui_tq_sk_dquoteindic, caihui_tq_sk_finindic, caihui_tq_sk_sharestruchg
from trade_date import ATradeDate


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

    def __init__(self, index_id, reindex, look_back, period):

        self.index_id = index_id

        self.reindex_total = reindex
        self.look_back = look_back
        self.period = period
        self.reindex = self.reindex_total[self.look_back::self.period]

        self.df_index_historical_constituents = caihui_tq_ix_comp.load_index_historical_constituents(
            self.index_id,
            begin_date=self.reindex[0].strftime('%Y%m%d'),
            end_date=self.reindex[-1].strftime('%Y%m%d')
        )
        self.stock_pool = self.df_index_historical_constituents.loc[:, ['stock_id', 'stock_code']].drop_duplicates(subset=['stock_id']).set_index('stock_id').sort_index()

        self.df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_price(stock_ids=self.stock_pool.index, reindex=self.reindex_total)
        self.df_stock_ret = self.df_stock_prc.pct_change().iloc[1:]
        self.df_stock_status = self.__load_stock_status()

        self.df_stock_industry = caihui_tq_sk_basicinfo.load_stock_industry(stock_ids=self.stock_pool.index)
        self.df_stock_historical_share = self.__load_stock_historical_share()
        self.stock_market_data, self.stock_financial_data = self.__load_stock_data()
        # self.stock_market_data = self.stock_financial_data = None

        self.ser_index_nav = caihui_tq_qt_index.load_index_nav(index_ids=[self.index_id], reindex=self.reindex_total)[self.index_id]

    def __load_stock_status(self):

        engine = database.connection('caihui')
        metadata = MetaData(bind=engine)
        t = Table('tq_qt_skdailyprice', metadata, autoload=True)

        columns = [
            t.c.TRADEDATE.label('trade_date'),
            t.c.SECODE.label('stock_id'),
            t.c.LCLOSE.label('l_close'),
            t.c.TCLOSE.label('t_close'),
            t.c.VOL.label('vol')
        ]

        s = select(columns).where(and_(
            t.c.SECODE.in_(self.stock_pool.index),
            t.c.TRADEDATE>=self.reindex_total[0].strftime('%Y%m%d'),
            t.c.TRADEDATE<=self.reindex_total[-1].strftime('%Y%m%d')
        ))

        df = pd.read_sql(s, engine, index_col=['trade_date', 'stock_id'], parse_dates=['trade_date'])

        df_stock_status = df.apply(self.__status_algo, axis='columns').unstack().fillna(4)

        return df_stock_status

    def __status_algo(self, ser):

        if ser.loc['vol'] == 0:
            return 3
        elif round(ser.loc['l_close']*1.1, 2) <= ser.loc['t_close']:
            return 1
        elif round(ser.loc['l_close']*0.9, 2) >= ser.loc['t_close']:
            return 2
        else:
            return 0

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

    def __load_stock_data(self):

        df_stock_market_data = caihui_tq_sk_dquoteindic.load_stock_market_data(
            stock_ids=self.stock_pool.index,
            begin_date=self.reindex_total[0].strftime('%Y%m%d'),
            end_date=self.reindex_total[-1].strftime('%Y%m%d')
        )
        # stock_market_data = {col: df_stock_market_data[col].unstack(0) for col in df_stock_market_data.columns}
        df_stock_market_data = df_stock_market_data.unstack()

        df_stock_financial_data = caihui_tq_sk_finindic.load_stock_financial_data(
            stock_ids=self.stock_pool.index,
            begin_date=self.reindex_total[0].strftime('%Y%m%d'),
            end_date=self.reindex_total[-1].strftime('%Y%m%d')
        )
        # stock_financial_data = {col: df_stock_financial_data[col].unstack(0) for col in df_stock_financial_data.columns}
        df_stock_financial_data = df_stock_financial_data.unstack()

        return df_stock_market_data, df_stock_financial_data


class StockPortfolio: # (metaclass=MetaClassPropertyDecorator)

    _ref_list = {}

    # _name_list_in_data = [
        # 'index_id',
        # 'reindex_total',
        # 'look_back',
        # 'period',
        # 'reindex',
        # 'df_index_historical_constituents',
        # 'stock_pool',
        # 'df_stock_prc',
        # 'df_stock_ret',
        # 'df_stock_status',
        # 'df_stock_industry',
        # 'df_stock_historical_share',
        # 'stock_market_data',
        # 'stock_financial_data',
        # 'ser_index_nav'
    # ]

    # _name_list_private = [
        # 'df_stock_pos',
        # 'df_stock_pos_adjusted',
        # 'ser_portfolio_nav',
        # 'ser_portfolio_inc',
        # 'ser_turnover'
    # ]

    def __init__(self, index_id, reindex, look_back, period, *args, **kwargs):

        ref = f'{index_id}, {reindex[0]}, {reindex[-1]}, {reindex.size}, {look_back}, {period}'
        sha1 = hashlib.sha1()
        sha1.update(ref.encode('utf-8'))
        ref = sha1.hexdigest()

        if ref in StockPortfolio._ref_list:
            self._data = StockPortfolio._ref_list[ref]
        else:
            self._data = StockPortfolio._ref_list[ref] = StockPortfolioData(index_id, reindex, look_back, period)

        self._df_stock_pos = None
        self._df_stock_pos_adjusted = None

        self._ser_portfolio_nav = None
        self._ser_portfolio_inc = None
        self._ser_turnover = None

    @property
    def index_id(self):

        return self._data.index_id

    @property
    def reindex_total(self):

        return self._data.reindex_total

    @property
    def look_back(self):

        return self._data.look_back

    @property
    def period(self):

        return self._data.period

    @property
    def reindex(self):

        return self._data.reindex

    @property
    def df_index_historical_constituents(self):

        return self._data.df_index_historical_constituents

    @property
    def stock_pool(self):

        return self._data.stock_pool

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

    def cal_portfolio_nav(self):

        if self.ser_portfolio_nav is not None:
            return self.ser_portfolio_nav

        self.cal_stock_pos_days()

        df_stock_pos_adjusted = pd.DataFrame(index=self.reindex, columns=self.stock_pool.index)
        df_stock_pos_adjusted.loc[self.reindex[0]] = self.df_stock_pos.loc[self.reindex[0], :].fillna(0.0)
        ser_portfolio_nav = pd.Series(1.0, index=self.reindex, name='nav')
        ser_portfolio_inc = pd.Series(0.0, index=self.reindex, name='inc')
        ser_turnover = pd.Series(0.0, index=self.reindex, name='turnover')

        for last_trade_date, trade_date in zip(self.reindex[:-1], self.reindex[1:]):

            stock_pos = (df_stock_pos_adjusted.loc[last_trade_date] * self.df_stock_prc.loc[trade_date] / self.df_stock_prc.loc[last_trade_date]).fillna(0.0)

            nav = stock_pos.sum()
            ser_portfolio_inc.loc[trade_date] = nav - 1.0
            ser_portfolio_nav.loc[trade_date] = ser_portfolio_nav.loc[last_trade_date] * nav

            stock_pos.loc[:] = stock_pos / nav
            stock_pos_standard = self.df_stock_pos.loc[trade_date].fillna(0.0)
            stock_status = self.df_stock_status.loc[trade_date]

            index_adjustable_stock = stock_status.loc[stock_status==0].index
            sum_pos_adjustable = 1.0 - stock_pos.loc[stock_status>0].sum()
            sum_pos_standard = 1.0 - stock_pos_standard.loc[stock_status>0].sum()

            stock_pos_adjusted = copy.deepcopy(stock_pos)
            stock_pos_adjusted.loc[index_adjustable_stock] = stock_pos_standard.loc[index_adjustable_stock] / sum_pos_standard * sum_pos_adjustable
            ser_turnover.loc[trade_date] = (stock_pos_adjusted - stock_pos).abs().sum()

            df_stock_pos_adjusted.loc[trade_date] = stock_pos_adjusted

        self._df_stock_pos_adjusted = df_stock_pos_adjusted
        self._ser_portfolio_nav = ser_portfolio_nav
        self._ser_portfolio_inc = ser_portfolio_inc
        self._ser_turnover = ser_turnover

        return ser_portfolio_nav

    def cal_stock_pos_days(self):

        if self.df_stock_pos is not None:
            return self.df_stock_pos

        ser_reindex = pd.Series(self.reindex, index=self.reindex)
        df_stock_pos = pd.DataFrame(index=self.reindex, columns=self.stock_pool.index)
        df_stock_pos.loc[:, :] = ser_reindex.apply(self.cal_stock_pos)

        self._df_stock_pos = df_stock_pos

        return df_stock_pos

    def cal_stock_pos(self, trade_date):

        raise NotImplementedError('Method \'cal_stock_pos\' is not defined.')

    def _load_stock_price(self, trade_date):

        reindex = self.reindex_total[self.reindex_total<=trade_date][-self.look_back-1:]
        stock_ids = self._load_stock_pool(trade_date).index

        # df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_prc(stock_ids=stock_ids, reindex=reindex)
        df_stock_prc = self.df_stock_prc.reindex(index=reindex, columns=stock_ids)

        return df_stock_prc

    def _load_stock_return(self, trade_date):

        reindex = self.reindex_total[self.reindex_total<=trade_date][-self.look_back:]
        stock_ids = self._load_stock_pool(trade_date).index

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

        self.cal_portfolio_nav()

        if reindex is None:
            reindex = self.reindex

        portfolio_return = self.ser_portfolio_nav.loc[reindex[-1]] / self.ser_portfolio_nav.loc[reindex[0]] - 1.0
        free_risk_rate = 0.0
        std_excess_return = self.ser_portfolio_inc.reindex(reindex).std()
        sharpe_ratio = (portfolio_return - free_risk_rate) / std_excess_return

        return portfolio_return, std_excess_return, sharpe_ratio


class StockPortfolioMarketCap(StockPortfolio):

    def __init__(self, index_id, reindex, look_back, period, *args, **kwargs):

        super(StockPortfolioMarketCap, self).__init__(index_id, reindex, look_back, period, *args, **kwargs)

    # Refrence: http://www.csindex.com.cn/zh-CN/indices/index-detail/000906
    def cal_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index
        df_stock_share = self._load_stock_share(trade_date)
        ser_stock_total_market_cap = self.stock_financial_data['total_market_cap'].loc[trade_date, stock_ids]

        ser_free_float_weight = df_stock_share.free_float_share / df_stock_share.total_share
        ser_free_float_weight.loc[:] = ser_free_float_weight.apply(self._weight_adjustment_algo)

        stock_pos = pd.Series(index=stock_ids, name=trade_date)
        stock_pos.loc[:] = (ser_stock_total_market_cap * ser_free_float_weight).fillna(0.0)
        stock_pos.loc[:] = stock_pos / stock_pos.sum()

        return stock_pos

    def _load_stock_share(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

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

    def __init__(self, index_id, reindex, look_back, period, *args, **kwargs):

        super(StockPortfolioEqualWeight, self).__init__(index_id, reindex, look_back, period, *args, **kwargs)

    def cal_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index

        stock_pos = pd.Series(1.0/stock_ids.size, index=stock_ids, name=trade_date)

        return stock_pos


class StockPortfolioMininumVolatility(StockPortfolio):

    def __init__(self, index_id, reindex, look_back, period, *args, **kwargs):

        super(StockPortfolioMininumVolatility, self).__init__(index_id, reindex, look_back, period, *args, **kwargs)

    def cal_stock_pos(self, trade_date):

        df_stock_ret = self._load_stock_return(trade_date)
        stock_ids = df_stock_ret.columns
        df_stock_status = self.df_stock_status.loc[df_stock_ret.index, stock_ids]

        df_stock_ret[df_stock_status>2] = np.nan
        ser_stock_volatility = df_stock_ret.std()

        stock_pos = pd.Series(index=stock_ids, name=trade_date)
        stock_pos.loc[:] = ser_stock_volatility.apply(lambda x: 1.0/x if x>0.0 else 0.0)
        stock_pos.loc[:] = stock_pos / stock_pos.sum()

        return stock_pos


class StockPortfolioMomentum(StockPortfolio):

    def __init__(self, index_id, reindex, look_back, period, exclusion, *args, **kwargs):

        super(StockPortfolioMomentum, self).__init__(index_id, reindex, look_back, period, *args, **kwargs)

        self._exclusion = exclusion

    @property
    def exclusion(self):

        return self._exclusion

    def cal_stock_pos(self, trade_date):

        df_stock_prc = self._load_stock_price(trade_date)
        stock_ids = df_stock_prc.columns

        ser_stock_momentum = df_stock_prc.iloc[-1-self.exclusion] / df_stock_prc.iloc[0]

        stock_pos = pd.Series(index=stock_ids, name=trade_date)
        stock_pos.loc[:] = (math.e ** ser_stock_momentum).fillna(0.0)
        stock_pos.loc[:] = stock_pos / stock_pos.sum()

        return stock_pos


class StockPortfolioSmallSize(StockPortfolioMarketCap):

    def __init__(self, index_id, reindex, look_back, period, percentage, *args, **kwargs):

        super(StockPortfolioSmallSize, self).__init__(index_id, reindex, look_back, period, *args, **kwargs)

        self._percentage = percentage

    @property
    def percentage(self):

        return self._percentage

    def cal_stock_pos(self, trade_date):

        stock_ids = self._load_stock_pool(trade_date).index
        df_stock_share = self._load_stock_share(trade_date)
        ser_stock_total_market_cap = self.stock_financial_data['total_market_cap'].loc[trade_date, stock_ids]

        ser_free_float_market_value = df_stock_share.free_float_share / df_stock_share.total_share * ser_stock_total_market_cap
        portfolio_size = round(stock_ids.size * self.percentage)
        small_size_stock_ids = ser_free_float_market_value.sort_values(ascending=True).iloc[:portfolio_size].index

        stock_pos = pd.Series(0.0, index=stock_ids, name=trade_date)
        stock_pos.loc[small_size_stock_ids] = 1.0 / portfolio_size

        return stock_pos


class StockPortfolioIndustry(StockPortfolio):

    def __init__(self, index_id, reindex, look_back, period, sw_industry_code, *args, **kwargs):

        super(StockPortfolioIndustry, self).__init__(index_id, reindex, look_back, period, *args, **kwargs)

        self._sw_industry_code = sw_industry_code

    @property
    def sw_industry_code(self):

        return self._sw_industry_code

    def _load_stock_pool(self, trade_date):

        stock_pool = super(StockPortfolioIndustry, self)._load_stock_pool(trade_date)

        stock_ids_by_industry = self._load_stock_ids_by_industry()
        stock_pool = stock_pool.reindex(stock_ids_by_industry)

        return stock_pool

    def _load_stock_ids_by_industry(self):

        stock_ids_by_industry = self.df_stock_industry.loc[self.df_stock_industry.sw_level1_code==self.sw_industry_code].index

        return stock_ids_by_industry


class StockPortfolioIndustryMininumVolatility(StockPortfolioIndustry, StockPortfolioMininumVolatility):

    def __init__(self, index_id, reindex, look_back, period, sw_industry_code, *args, **kwargs):

        super(StockPortfolioIndustryMininumVolatility, self).__init__(index_id, reindex, look_back, period, sw_industry_code, *args, **kwargs)


class StockPortfolioIndustryMomentum(StockPortfolioIndustry, StockPortfolioMomentum):

    def __init__(self, index_id, reindex, look_back, period, sw_industry_code, exclusion, *args, **kwargs):

        super(StockPortfolioIndustryMomentum, self).__init__(index_id, reindex, look_back, period, sw_industry_code, exclusion, *args, **kwargs)


def func(class_name, index_id, trade_dates, look_back, period, sw_industry_codes, *args, **kwargs):

    df = pd.DataFrame()

    for sw_industry_code in sw_industry_codes:

        if sw_industry_code is not None:

            cls = globals()[class_name]
            portfolio = cls(index_id, trade_dates, look_back, period, sw_industry_code, *args, **kwargs)
            df[sw_industry_code] = portfolio.cal_portfolio_nav()

            df.to_csv(f'{class_name}_{sw_industry_code}.csv')

def multiprocessing_cal_portfolio_nav_by_industry(class_name, index_id, trade_dates, look_back, period, cpu_count=4, *args, **kwargs):

    portfolio = StockPortfolio(index_id, trade_dates, look_back, period)

    arr_sw_industry_codes = np.array(portfolio.df_stock_industry.sw_level1_code.drop_duplicates().sort_values())

    if portfolio.stock_market_data is None and portfolio.stock_financial_data is None:
        arr_sw_industry_codes = arr_sw_industry_codes.reshape(-1, 1)
    else:
        warnings.warn('如果不使用stock_market_data和stock_financial_data, 请在StockPortfolioData中将其赋值为None.', Warning)
        arr_sw_industry_codes = np.append(arr_sw_industry_codes, [None] * (-arr_sw_industry_codes.size % cpu_count))
        arr_sw_industry_codes = arr_sw_industry_codes.reshape(cpu_count, -1)

    for sw_industry_codes in arr_sw_industry_codes:

        process = multiprocessing.Process(target=func, args=(class_name, index_id, trade_dates, look_back, period, sw_industry_codes, *args), kwargs={**kwargs})
        process.start()


if __name__ == '__main__':

    index_id = '2070000191'
    begin_date = '2019-01-01'
    end_date = '2019-03-08'
    look_back = 120
    period = 1

    # bug: look_back == 0
    trade_dates = ATradeDate.trade_date(begin_date=begin_date, end_date=end_date, lookback=look_back)
    trade_dates.name = 'trade_date'

    dict_portfolio = {}
    df_portfolio_nav = pd.DataFrame()

    # dict_portfolio['MarketCap'] = StockPortfolioMarketCap(index_id, trade_dates, look_back, period)
    # df_portfolio_nav['MarketCap'] = dict_portfolio['MarketCap'].cal_portfolio_nav()

    # dict_portfolio['EqualWeight'] = StockPortfolioEqualWeight(index_id, trade_dates, look_back, period)
    # df_portfolio_nav['EqualWeight'] = dict_portfolio['EqualWeight'].cal_portfolio_nav()

    # dict_portfolio['MininumVolatility'] = StockPortfolioMininumVolatility(index_id, trade_dates, look_back, period)
    # df_portfolio_nav['MininumVolatility'] = dict_portfolio['MininumVolatility'].cal_portfolio_nav()

    # dict_portfolio['Momentum'] = StockPortfolioMomentum(index_id, trade_dates, look_back, period, exclusion=20)
    # df_portfolio_nav['Momentum'] = dict_portfolio['Momentum'].cal_portfolio_nav()

    # dict_portfolio['SmallSize'] = StockPortfolioSmallSize(index_id, trade_dates, look_back, period, percentage=0.3)
    # df_portfolio_nav['SmallSize'] = dict_portfolio['SmallSize'].cal_portfolio_nav()

    # df_portfolio_nav.to_csv('df_portfolio_nav.csv')
    # set_trace()

    # multiprocessing_cal_portfolio_nav_by_industry('StockPortfolioIndustryMininumVolatility', index_id, trade_dates, look_back, period)

    # multiprocessing_cal_portfolio_nav_by_industry('StockPortfolioIndustryMomentum', index_id, trade_dates, look_back, period, exclusion=30)
    # set_trace()

