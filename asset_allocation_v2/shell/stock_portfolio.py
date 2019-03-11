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
sys.path.append('shell')
from db import database
from db import caihui_tq_ix_comp, caihui_tq_sk_dquoteindic
from trade_date import ATradeDate
from ipdb import set_trace


logger = logging.getLogger(__name__)


class StockPortfolio(object):

    def __init__(self, index_id, reindex=None, look_back=None, period=1):

        self.__index_id = index_id

        self.__reindex_total = reindex
        self.__look_back = look_back
        self.__period = period
        self.__reindex = self.reindex_total[self.look_back-1::self.period]

        stock_pool = caihui_tq_ix_comp.load_index_historical_constituents(
            self.index_id,
            begin_date=self.reindex[0].strftime('%Y%m%d'),
            end_date=self.reindex[-1].strftime('%Y%m%d')
        )
        stock_pool = stock_pool.loc[:, ['stock_id', 'stock_code']].drop_duplicates(['stock_id']).set_index('stock_id').sort_index()
        self.__stock_pool = stock_pool

        self.__df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_price(stock_ids=self.stock_pool.index, reindex=self.reindex_total)
        self.__df_stock_ret = self.df_stock_prc.pct_change().iloc[1:]

        self.__df_stock_status = None
        self.__df_stock_pos = None
        self.__df_stock_pos_adjusted = None
        self.__ser_portfolio_nav = None
        self.__ser_portfolio_inc = None

    @property
    def index_id(self):

        return self.__index_id

    @property
    def reindex_total(self):

        return self.__reindex_total

    @property
    def look_back(self):

        return self.__look_back

    @property
    def period(self):

        return self.__period

    @property
    def pos(self):

        return self.__pos

    @property
    def reindex(self):

        return self.__reindex

    @property
    def stock_pool(self):

        return self.__stock_pool

    @property
    def df_stock_prc(self):

        return self.__df_stock_prc

    @property
    def df_stock_ret(self):

        return self.__df_stock_ret

    @property
    def df_stock_status(self):

        return self.__df_stock_status

    @property
    def df_stock_pos(self):

        return self.__df_stock_pos

    @property
    def df_stock_pos_adjusted(self):

        return self.__df_stock_pos_adjusted

    @property
    def ser_portfolio_nav(self):

        return self.__ser_portfolio_nav

    @property
    def ser_portfolio_inc(self):

        return self.__ser_portfolio_inc

    def _load_stock_ret(self, date):

        reindex = self.reindex_total[self.reindex_total<=date][1-self.look_back:]
        stock_pool = caihui_tq_ix_comp.load_index_constituents(self.index_id, date.strftime('%Y%m%d'))

        # df_stock_prc = caihui_tq_sk_dquoteindic.load_stock_prc(stock_ids=stock_pool.index, reindex=reindex)
        # df_stock_ret = df_stock_prc.pct_change().iloc[1:]

        df_stock_ret = self.df_stock_ret.reindex(index=reindex, columns=stock_pool.index)

        return df_stock_ret

    def cal_stock_pos(self, date):

        df_stock_ret = self._load_stock_ret(date)
        stock_ids = df_stock_ret.columns
        pos = pd.DataFrame(1/800, index=[date], columns=stock_ids)

        return pos

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

            for date in bar:

                logger.debug("%s : %s", s, date.strftime('%Y-%m-%d'))

                pos = self.cal_stock_pos(date)
                df_stock_pos = pd.concat([df_stock_pos, pos], sort=True)

        self.__df_stock_pos = df_stock_pos

        return df_stock_pos

    def _func(self, ser):

        if round(ser.loc['l_close']*1.1, 2) <= ser.loc['t_close']:
            return 1
        elif round(ser.loc['l_close']*0.9, 2) >= ser.loc['t_close']:
            return 2
        if ser.loc['vol'] == 0:
            return 3
        else:
            return 0

    def load_stock_status(self):

        if self.df_stock_status is not None:
            return self.df_stock_status

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
            t.c.TRADEDATE>=self.reindex[0].strftime('%Y%m%d'),
            t.c.TRADEDATE<=self.reindex[-1].strftime('%Y%m%d'),
            t.c.SECODE.in_(self.stock_pool.index)
        ))

        df = pd.read_sql(s, engine, index_col=['trade_date', 'stock_id'], parse_dates=['trade_date'])

        df_stock_status = df.apply(self._func, axis='columns').unstack().fillna(4)

        self.__df_stock_status = df_stock_status

        return df_stock_status

    def cal_portfolio_nav(self):

        if self.ser_portfolio_nav is not None:
            return self.ser_portfolio_nav

        self.cal_stock_pos_days()
        self.load_stock_status()

        df_stock_pos_adjusted = pd.DataFrame(columns=self.stock_pool.index)
        df_stock_pos_adjusted = pd.concat([df_stock_pos_adjusted, self.df_stock_pos.iloc[[0], :].fillna(0.0)], sort=True)

        ser_portfolio_nav = pd.Series(1.0, index=self.reindex)
        ser_portfolio_inc = pd.Series(0.0, index=self.reindex)

        for last_trade_date, trade_date in zip(self.reindex[:-1], self.reindex[1:]):

            stock_pos_adjusted = df_stock_pos_adjusted.loc[last_trade_date] * (self.df_stock_ret.loc[trade_date] + 1.0).fillna(0.0)

            sum_ = stock_pos_adjusted.sum()
            ser_portfolio_inc.loc[trade_date] = sum_ - 1.0
            ser_portfolio_nav.loc[trade_date] = ser_portfolio_nav.loc[last_trade_date] * sum_

            stock_pos_adjusted.loc[:] = stock_pos_adjusted / sum_
            stock_pos = self.df_stock_pos.loc[trade_date].fillna(0.0)
            sum_adjustable_pos = 1.0
            sum_pos = 1.0
            set_adjustable_stock = set()

            for stock_id in self.stock_pool.index:

                if self.df_stock_status.loc[trade_date, stock_id] in [1, 3, 4] and stock_pos_adjusted.loc[stock_id] < stock_pos.loc[stock_id]:
                    sum_adjustable_pos -= stock_pos_adjusted.loc[stock_id]
                    sum_pos -= stock_pos.loc[stock_id]

                elif self.df_stock_status.loc[trade_date, stock_id] in [2, 3, 4] and stock_pos_adjusted.loc[stock_id] > stock_pos.loc[stock_id]:
                    sum_adjustable_pos -= stock_pos_adjusted.loc[stock_id]
                    sum_pos -= stock_pos.loc[stock_id]

                else:
                    set_adjustable_stock.add(stock_id)

            index_adjustable_stock = pd.Index(set_adjustable_stock)
            stock_pos_adjusted.loc[index_adjustable_stock] = stock_pos.loc[index_adjustable_stock] / sum_pos * sum_adjustable_pos

            df_stock_pos_adjusted = pd.concat([df_stock_pos_adjusted, stock_pos_adjusted.to_frame(name=trade_date).T], sort=True)

        self.__df_stock_pos_adjusted = df_stock_pos_adjusted
        self.__ser_portfolio_nav = ser_portfolio_nav
        self.__ser_portfolio_inc = ser_portfolio_inc

        return ser_portfolio_nav

    def portfolio_analysis(self, reindex=None):

        self.cal_portfolio_nav()

        if reindex is None:
            reindex = self.reindex

        portfolio_return = self.ser_portfolio_nav.loc[reindex[-1]] / self.ser_portfolio_nav.loc[reindex[0]]
        free_risk_rate = 0
        std_excess_return = self.ser_portfolio_inc.reindex(reindex).std()
        sharpe_ratio = (portfolio_return - free_risk_rate) / std_excess_return

        return ret, risk, sharpe_ratio


if __name__ == '__main__':

    begin_date = '2018-07-27'
    end_date = '2019-03-08'
    look_back = 120

    trade_date = ATradeDate.trade_date(begin_date=begin_date, end_date=end_date, lookback=look_back)

    portfolio = StockPortfolio('2070000191', reindex=trade_date, look_back=look_back)
    # df = portfolio.cal_stock_pos_days()
    # df2 = portfolio.load_df_status()
    ser = portfolio.cal_portfolio_nav()
    set_trace()

