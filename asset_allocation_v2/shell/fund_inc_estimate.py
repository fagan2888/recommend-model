#coding=utf-8
'''
Created at Dec. 28, 2018
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import sys
import logging
sys.path.append('shell')
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Text, Integer, Float, Date, DateTime, ForeignKey
from sqlalchemy import MetaData, Table, select, func, and_, not_, tuple_
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ipdb import set_trace
from db import database, base_ra_fund_nav
from db import caihui_tq_fd_basicinfo, caihui_tq_fd_skdetail, caihui_tq_sk_dquoteindic, caihui_tq_qt_index
from trade_date import ATradeDate
from util_timestamp import *


logger = logging.getLogger(__name__)

Base = declarative_base()

class FundIncEstimate(object):

    def __init__(self, fund_ids=None, fund_codes=None, begin_date=None, end_date=None):

        if end_date is None:
            self._end_date = pd.Period.now(freq="D").start_time
            if  pd.Timestamp.now().hour < 18:
                self._end_date = trade_date_before(self._end_date)
        else:
            self._end_date = pd.Timestamp(end_date)

        if begin_date is None:
            self._begin_date = trade_date_not_later_than(self._end_date)
        else:
            self._begin_date = pd.Timestamp(begin_date)

        if fund_codes is not None or fund_ids is not None:
            self._fund_pool = caihui_tq_fd_skdetail.load_fund_info(fund_codes=fund_codes, fund_ids=fund_ids)
        else:
            self._fund_pool = load_fund_pool(self, self._end_date)

        if self._fund_pool.empty:
            print('EMPTY!')
            set_trace()

    @property
    def fund_pool(self):
        return self._fund_pool

    @property
    def begin_date(self):
        return self._begin_date

    @property
    def end_date(self):
        return self._end_date

    @staticmethod
    def load_fund_pool(date):

        db = database.connection('caihui')
        metadata = MetaData(bind=db)
        t = Table('tq_fd_basicinfo', metadata, autoload=True)

        columns = [
                t.c.SECODE.label('fund_id'),
                t.c.FSYMBOL.label('fund_code')
        ]

        s = select(columns).where(and_(
                t.c.FDSTYLE.in_([2, 3, 4]),
                t.c.FOUNDDATE!='19000101',
                t.c.FOUNDDATE<=last_year(date).strftime('%Y%m%d'),
                t.c.ENDDATE=='19000101',
                t.c.FDNATURE.in_(['证券投资基金', 'LOF']),
                t.c.FDMETHOD=='开放式基金',
                t.c.TOTSHARE>0.1,
                not_(t.c.FDNAME.contains('联接')),
                not_(t.c.FDNAME.contains('沪港深')),
                not_(t.c.FDNAME.contains('港股通'))
        ))

        df = pd.read_sql(s, db, index_col=['fund_id'])

        return df

set_trace()

class fund_inc_est_sk_pos(fundIncEstimate):

    def __init__(self, fund_ids=None, fund_codes=None, begin_date=None, end_date=None):
        super(fund_inc_est_sk_pos, self).__init__(fund_ids=None, fund_codes=None, begin_date=None, end_date=None)



    def cal_sk_pos(self, fund_stock_pos, fund_stock_pos_ten):

        fund_stock_pos_new = pd.DataFrame()
        for fund_id in fund_stock_pos.index.levels[0]:

            stock_pos = fund_stock_pos.loc[fund_id]
            stock_pos.index.name = 'stock_id'

            if fund_id in fund_stock_pos_ten.index.levels[0]:

                stock_pos_ten = fund_stock_pos_ten.loc[fund_id]
                stock_pos_ten.index.name = 'stock_id'

                min_navrto = stock_pos_ten.navrto.min()
                stock_pos[stock_pos.navrto > min_navrto] = min_navrto
                stock_pos = stock_pos.drop(stock_pos_ten.index.intersection(stock_pos.index))
                stock_pos = pd.concat([stock_pos, stock_pos_ten])

            stock_pos = stock_pos.reset_index()
            stock_pos['fund_id'] = fund_id

            fund_stock_pos_new = pd.concat([fund_stock_pos_new, stock_pos])

        for fund_id in fund_stock_pos_ten.index.levels[0].difference(fund_stock_pos.index.levels[0]):

            stock_pos_ten = fund_stock_pos_ten.loc[fund_id]
            stock_pos_ten.index.name = 'stock_id'
            stock_pos_ten = stock_pos_ten.reset_index()
            stock_pos_ten['fund_id'] = fund_id

            fund_stock_pos_new = pd.concat([fund_stock_pos_new, stock_pos_ten])

        fund_stock_pos_new = fund_stock_pos_new.set_index(['fund_id', 'stock_id']).sort_index()

        return fund_stock_pos_new


    def cal_sk_pos_days(self):

        fund_ids = self._fund_pool.index

        if self._begin_date.month < 4:
            begin_date = pd.Timestamp(self._begin_date.year-1, 9, 1)
        elif self._begin_date.month < 9:
            begin_date = pd.Timestamp(self._begin_date.year, 4, 1)
        else:
            begin_date = pd.Timestamp(self._begin_date.year, 9, 1)
        dates = pd.Series(index=[begin_date, end_date])
        dates = dates.resample('MS').first().index

        df_fund_stock_pos = pd.DataFrame()
        for date in dates:
            print('cal_sk_pos_days', date)
            if date.month == 4:
                fund_stock_pos = load_all_skdetail(self, fund_ids, pd.Timestamp(date.year-1, 12, 31)).dropna()
            elif date.month == 5:
                fund_stock_pos_ten = load_ten_skdetail(self, fund_ids, pd.Timestamp(date.year, 3, 31)).dropna()
            elif date.month == 8:
                fund_stock_pos_ten = load_ten_skdetail(self, fund_ids, pd.Timestamp(date.year, 6, 30)).dropna()
            elif date.month == 9:
                fund_stock_pos = load_all_skdetail(self, fund_ids, pd.Timestamp(date.year, 6, 30)).dropna()
            elif date.month == 11:
                fund_stock_pos_ten = load_ten_skdetail(self, fund_ids, pd.Timestamp(date.year, 9, 30)).dropna()
            elif date.month == 2:
                fund_stock_pos_ten = load_ten_skdetail(self, fund_ids, pd.Timestamp(date.year-1, 12, 31)).dropna()

            if date.month in [2, 5, 8, 11]:
                fund_stock_pos = cal_sk_pos(self, fund_stock_pos, fund_stock_pos_ten)

            fund_stock_pos['date'] = date
            df_fund_stock_pos = pd.concat([df_fund_stock_pos, fund_stock_pos])
            fund_stock_pos = fund_stock_pos.drop('date', axis='columns')

        df_fund_stock_pos = df_fund_stock_pos.reset_index().set_index(['date', 'fund_id', 'stock_id']) / 100.0

        return df_fund_stock_pos


    def estimate_fund_inc(self):

        df_stock_nav = caihui_tq_sk_dquoteindic.load_stock_nav_daily(
                begin_date=trade_date_before(self._begin_date).strftime('%Y%m%d'),
                end_date=self._end_date.strftime('%Y%m%d')
        ).dropna(how='all', axis='columns')
        df_stock_ret = df_stock_nav.pct_change().iloc[1:]

        df_fund_stock_pos = cal_sk_pos_days(self)

        dates = []
        date = self._begin_date
        while date <= self._end_date:
            dates.append(date)
            date = start_of_month(next_month(date))
        dates.append(self._end_date + pd.Timedelta('1d'))

        df_fund_inc_est = pd.DataFrame()
        for date, pdate in zip(dates[:-1], dates[1:]):
            print('fund_ret_est_sk_pos', date, pdate)
            pdate -= pd.Timedelta('1d')

            fund_stock_pos = df_fund_stock_pos.loc[start_of_month(date)]
            fund_stock_pos = fund_stock_pos.unstack(level=0).navrto.fillna(0.0)
            stock_ret = df_stock_ret.loc[date:pdate].reindex(fund_stock_pos.index, axis='columns').fillna(0.0)

            fund_inc_est = pd.DataFrame(
                    np.dot(stock_ret, fund_stock_pos),
                    index=stock_ret.index,
                    columns=fund_stock_pos.columns
            )
            df_fund_inc_est = pd.concat([df_fund_inc_est, fund_inc_est])

        df_fund_inc_est = df_fund_inc_est.rename(lambda x: fund_pool.fund_code.loc[x], axis='columns').sort_index(axis='columns')
        df_fund_inc_est.columns.name = 'fund_code'

        return df_fund_inc_est



set_trace()
