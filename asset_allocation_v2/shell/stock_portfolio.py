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
import numpy as np
import pandas as pd
sys.path.append('shell')
from db import caihui_tq_ix_comp, caihui_tq_sk_dquoteindic
from trade_date import ATradeDate
from ipdb import set_trace


logger = logging.getLogger(__name__)


class Portfolio(object):

    def __init__(self, index_id, reindex=None, look_back=None, period=1):

        self.__index_id = index_id

        self.__reindex_total = reindex
        self.__look_back = look_back
        self.__period = period
        self.__reindex = self.reindex_total[self.look_back-1::self.period]

        stock_pool = caihui_tq_ix_comp.load_index_historical_constituents(self.index_id, begin_date=self.reindex[0], end_date=self.reindex[-1])
        stock_pool = stock_pool.loc[:, ['stock_id', 'stock_code']].drop_duplicates(['stock_id']).set_index('stock_id')
        self.__stock_pool = stock_pool

        self.__df_stock_nav = caihui_tq_sk_dquoteindic.load_stock_nav(stock_ids=self.stock_pool.index, reindex=self.reindex_total)
        self.__df_stock_ret = self.df_stock_nav.pct_change().iloc[1:]

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
    def df_stock_nav(self):

        return self.__df_stock_nav

    @property
    def df_stock_ret(self):

        return self.__df_stock_ret

    def load_stock_ret(self, date):

        reindex = self.reindex_total[self.reindex_total<=date][1-self.look_back:]
        stock_pool = caihui_tq_ix_comp.load_index_constituents(self.index_id, date)

        # df_stock_nav = caihui_tq_sk_dquoteindic.load_stock_nav(stock_ids=stock_pool.index, reindex=reindex)
        # df_stock_ret = df_stock_nav.pct_change().iloc[1:]

        df_stock_ret = self.df_stock_ret.reindex(index=reindex, columns=stock_pool.index)

        return df_stock_ret

    def cal_stock_pos(self, date, df_stock_ret):

        stock_ids = df_stock_ret.columns
        ws = pd.Series(1/800, index=stock_ids, name=date)

        return ws

    def cal_stock_pos_days(self):

        df_pos = pd.DataFrame(index=self.reindex)

        s = 'perform %-12s' % self.__class__.__name__

        with click.progressbar(
            self.reindex,
            label=s.ljust(30),
            item_show_func=lambda x: x.strftime("%Y-%m-%d") if x else None
        ) as bar:

            for date in bar:

                logger.debug("%s : %s", s, date.strftime("%Y-%m-%d"))

                df_stock_ret = self.load_stock_ret(date)
                ws = self.allocate_algo(date, df_stock_ret)
                df_pos.loc[date] = ws

        return df_pos


if __name__ == '__main__':

    begin_date = '2012-07-27'
    end_date = '2019-03-08'
    look_back = 120

    trade_date = ATradeDate.trade_date(begin_date=begin_date, end_date=end_date, lookback=look_back)

    portfolio = Portfolio('2070000191', reindex=trade_date, look_back=look_back)
    df = portfolio.cal_stock_pos_days()

