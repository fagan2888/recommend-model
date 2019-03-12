#coding=utf-8
'''
Created on: Mar. 11, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import sys
import logging
import click
import warnings
from sqlalchemy import MetaData, Table, select, func, and_
import numpy as np
import pandas as pd
import hashlib
import copy
sys.path.append('shell')
from db import database
from db import caihui_tq_ix_comp, caihui_tq_sk_dquoteindic
from trade_date import ATradeDate
from ipdb import set_trace


logger = logging.getLogger(__name__)


class MetaClassPropertyDecorator(type):

    def __new__(cls, name, bases, attrs):

        attrs.update({name: property(MetaClassPropertyDecorator.generate_property_func_in_data(name)) for name in attrs.get('_name_list_in_data', [])})
        attrs.update({name: property(MetaClassPropertyDecorator.generate_property_func_private(name)) for name in attrs.get('_name_list_private', [])})

        return type.__new__(cls, name, bases, attrs)

    @staticmethod
    def generate_property_func_in_data(name):

        def func(self):

            if not hasattr(self, '_data'):
                raise ValueError
            else:
                data = getattr(self, '_data')
                if not hasattr(data, name):
                    raise ValueError

            return getattr(data, name)

        return func

    @staticmethod
    def generate_property_func_private(name):

        def func(self):

            if not hasattr(self, f'_{name}'):
                raise ValueError

            return getattr(self, f'_{name}')

        return func


class StockPortfolioData(object):

    def __init__(self, index_id, reindex, look_back, period):

        self.index_id = index_id

        self.reindex_total = reindex
        self.look_back = look_back
        self.period = period
        self.reindex = self.reindex_total[self.look_back-1::self.period]

        stock_pool = caihui_tq_ix_comp.load_index_historical_constituents(
            self.index_id,
            begin_date=self.reindex[0].strftime('%Y%m%d'),
            end_date=self.reindex[-1].strftime('%Y%m%d')
        )
        self.df_index_historical_constituents = copy.deepcopy(stock_pool)
        self.stock_pool = stock_pool.loc[:, ['stock_id', 'stock_code']].drop_duplicates(['stock_id']).set_index('stock_id').sort_index()

        self.df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_price(stock_ids=self.stock_pool.index, reindex=self.reindex_total)
        self.df_stock_ret = self.df_stock_prc.pct_change().iloc[1:]

        self.df_stock_status = self.__load_stock_status()

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

        df_stock_status = df.apply(self.__func, axis='columns').unstack().fillna(4)

        return df_stock_status

    def __func(self, ser):

        if ser.loc['vol'] == 0:
            return 3
        elif round(ser.loc['l_close']*1.1, 2) <= ser.loc['t_close']:
            return 1
        elif round(ser.loc['l_close']*0.9, 2) >= ser.loc['t_close']:
            return 2
        else:
            return 0


class StockPortfolio(object, metaclass=MetaClassPropertyDecorator):

    _ref_list = {}

    _name_list_in_data = [
        'index_id',
        'reindex_total',
        'look_back',
        'period',
        'reindex',
        'df_index_historical_constituents',
        'stock_pool',
        'df_stock_prc',
        'df_stock_ret',
        'df_stock_status'
    ]

    _name_list_private = [
        'df_stock_pos',
        'df_stock_pos_adjusted',
        'ser_portfolio_nav',
        'ser_portfolio_inc',
        'turnover'
    ]

    def __init__(self, index_id, reindex, look_back, period=1):

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
        self._turnover = None

    def cal_portfolio_nav(self):

        if self.ser_portfolio_nav is not None:
            return self.ser_portfolio_nav

        self.cal_stock_pos_days()

        df_stock_pos_adjusted = pd.DataFrame(columns=self.stock_pool.index)
        df_stock_pos_adjusted = pd.concat([df_stock_pos_adjusted, self.df_stock_pos.iloc[[0], :].fillna(0.0)], sort=True)
        ser_portfolio_nav = pd.Series(1.0, index=self.reindex)
        ser_portfolio_inc = pd.Series(0.0, index=self.reindex)
        turnover = 0.0

        s = 'perform %-12s' % self.__class__.__name__

        with click.progressbar(
            zip(self.reindex[:-1], self.reindex[1:]),
            label=s.ljust(30),
            item_show_func=lambda x: x[1].strftime('%Y-%m-%d') if x else None
        ) as bar:

            for last_trade_date, trade_date in bar:

                stock_pos = df_stock_pos_adjusted.loc[last_trade_date] * (self.df_stock_ret.loc[trade_date] + 1.0).fillna(0.0)

                nav = stock_pos.sum()
                ser_portfolio_inc.loc[trade_date] = nav - 1.0
                ser_portfolio_nav.loc[trade_date] = ser_portfolio_nav.loc[last_trade_date] * nav

                stock_pos.loc[:] = stock_pos / nav
                stock_pos_standard = self.df_stock_pos.loc[trade_date].fillna(0.0)
                sum_pos_adjustable = 1.0
                sum_pos_standard = 1.0
                list_adjustable_stock = []

                for stock_id in self.stock_pool.index:

                    # if self.df_stock_status.loc[trade_date, stock_id] in [1, 3, 4] and stock_pos.loc[stock_id] < stock_pos_standard.loc[stock_id]:
                        # sum_pos_adjustable -= stock_pos.loc[stock_id]
                        # sum_pos_standard -= stock_pos_standard.loc[stock_id]

                    # elif self.df_stock_status.loc[trade_date, stock_id] in [2, 3, 4] and stock_pos.loc[stock_id] > stock_pos_standard.loc[stock_id]:
                        # sum_pos_adjustable -= stock_pos.loc[stock_id]
                        # sum_pos_standard -= stock_pos_standard.loc[stock_id]

                    if self.df_stock_status.loc[trade_date, stock_id] > 0:
                        sum_pos_adjustable -= stock_pos.loc[stock_id]
                        sum_pos_standard -= stock_pos_standard.loc[stock_id]

                    else:
                        list_adjustable_stock.append(stock_id)

                index_adjustable_stock = pd.Index(list_adjustable_stock)
                stock_pos_adjusted = copy.deepcopy(stock_pos)
                stock_pos_adjusted.loc[index_adjustable_stock] = stock_pos_standard.loc[index_adjustable_stock] / sum_pos_standard * sum_pos_adjustable
                turnover += (stock_pos_adjusted - stock_pos).abs().sum()

                stock_pos_adjusted = stock_pos_adjusted.to_frame(name=trade_date).T
                df_stock_pos_adjusted = pd.concat([df_stock_pos_adjusted, stock_pos_adjusted], sort=True)

        self._df_stock_pos_adjusted = df_stock_pos_adjusted
        self._ser_portfolio_nav = ser_portfolio_nav
        self._ser_portfolio_inc = ser_portfolio_inc
        self._turnover = turnover

        return ser_portfolio_nav

    def cal_stock_pos_days(self):

        if self.df_stock_pos is not None:
            return self.df_stock_pos

        df_stock_pos = pd.DataFrame(columns=self.stock_pool.index)

        s = 'perform %-12s' % self.__class__.__name__

        with click.progressbar(
            self.reindex,
            label=s.ljust(30),
            item_show_func=lambda x: x.strftime('%Y-%m-%d') if x else None
        ) as bar:

            for trade_date in bar:

                logger.debug("%s : %s", s, trade_date.strftime('%Y-%m-%d'))

                stock_pos = self.cal_stock_pos(trade_date)
                df_stock_pos = pd.concat([df_stock_pos, stock_pos], sort=True)

        self._df_stock_pos = df_stock_pos

        return df_stock_pos

    def cal_stock_pos(self, trade_date):

        raise NotImplementedError('Method \'cal_stock_pos\' is not defined.')

    def _load_stock_ret(self, trade_date):

        reindex = self.reindex_total[self.reindex_total<=trade_date][1-self.look_back:]
        # stock_pool = caihui_tq_ix_comp.load_index_constituents(self.index_id, trade_date.strftime('%Y%m%d'))
        stock_pool = self.df_index_historical_constituents.loc[
            (self.df_index_historical_constituents.selected_date<=trade_date) & \
            ((self.df_index_historical_constituents.out_date>trade_date) | \
            (self.df_index_historical_constituents.out_date=='19000101'))
        ].loc[:, ['stock_id', 'stock_code']].set_index('stock_id').sort_index()

        # df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_prc(stock_ids=stock_pool.index, reindex=reindex)
        # df_stock_ret = df_stock_prc.pct_change().iloc[1:]
        df_stock_ret = self.df_stock_ret.reindex(index=reindex, columns=stock_pool.index)

        return df_stock_ret

    def portfolio_analysis(self, reindex=None):

        self.cal_portfolio_nav()

        if reindex is None:
            reindex = self.reindex

        portfolio_return = self.ser_portfolio_nav.loc[reindex[-1]] / self.ser_portfolio_nav.loc[reindex[0]] - 1.0
        free_risk_rate = 0.0
        std_excess_return = self.ser_portfolio_inc.reindex(reindex).std()
        sharpe_ratio = (portfolio_return - free_risk_rate) / std_excess_return

        return portfolio_return, std_excess_return, sharpe_ratio


class StockPortfolioEqualWeight(StockPortfolio):

    def __init__(self, index_id, reindex, look_back, period=1):

        super(StockPortfolioEqualWeight, self).__init__(index_id, reindex, look_back, period)

    def cal_stock_pos(self, trade_date):

        df_stock_ret = self._load_stock_ret(trade_date)
        stock_ids = df_stock_ret.columns

        stock_pos = pd.DataFrame(1.0/stock_ids.size, index=[trade_date], columns=stock_ids)

        return stock_pos


class StockPortfolioMininumVolatility(StockPortfolio):

    def __init__(self, index_id, reindex, look_back, period=1):

        super(StockPortfolioMininumVolatility, self).__init__(index_id, reindex, look_back, period)

    def cal_stock_pos(self, trade_date):

        df_stock_ret = self._load_stock_ret(trade_date)
        df_stock_status = self.df_stock_status.loc[df_stock_ret.index, df_stock_ret.columns]

        df_stock_ret[df_stock_status>2] = np.nan
        ser_stock_volatility = df_stock_ret.apply(np.std)
        stock_pos = ser_stock_volatility.apply(lambda x: 1.0/x if x>0.0 else 0.0)
        stock_pos.loc[:] = stock_pos / stock_pos.sum()

        stock_pos = stock_pos.to_frame(name=trade_date).T

        return stock_pos


# unfinished
class StockPortfolioSmallSize(StockPortfolio):

    def __init__(self, index_id, reindex, look_back, period=1):

        super(StockPortfolioSmallSize, self).__init__(index_id, reindex, look_back, period)

        self._df_a = self.__load_stock_data()

    def cal_stock_pos(self, trade_date):

        def_stock_ret = self._load_stock_ret(trade_date)

        return stock_pos


if __name__ == '__main__':

    begin_date = '2006-01-01'
    end_date = '2019-03-08'
    look_back = 120

    # bug: look_back == 0
    trade_dates = ATradeDate.trade_date(begin_date=begin_date, end_date=end_date, lookback=look_back)
    trade_dates.name = 'trade_date'

    portfolio = StockPortfolioEqualWeight('2070000191', reindex=trade_dates, look_back=look_back)
    ser = portfolio.cal_portfolio_nav()

    portfolio2 = StockPortfolioMininumVolatility('2070000191', reindex=trade_dates, look_back=look_back)
    ser2 = portfolio2.cal_portfolio_nav()
    set_trace()

