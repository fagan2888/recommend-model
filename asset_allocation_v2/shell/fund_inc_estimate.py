#coding=utf-8
'''
Created at Dec. 28, 2018
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import sys
import logging
from sqlalchemy import MetaData, Table, select, func, and_, not_
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ipdb import set_trace
import time
sys.path.append('shell')
from db import database, base_ra_fund_nav
from db import caihui_tq_fd_basicinfo, caihui_tq_fd_skdetail, caihui_tq_sk_dquoteindic, caihui_tq_ix_basicinfo, caihui_tq_qt_index
from trade_date import ATradeDate
from util_timestamp import *


logger = logging.getLogger(__name__)


class FundIncEstimate(object, metaclass = ABCMeta):

    def __init__(self, begin_date=None, end_date=None, fund_ids=None, fund_codes=None):

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

        if fund_codes is None and fund_ids is None:
            self._fund_pool = self.load_fund_pool(self._end_date)
        else:
            self._fund_pool = caihui_tq_fd_basicinfo.load_fund_basic_info(fund_codes=fund_codes, fund_ids=fund_ids)

        if self._fund_pool.empty:
            print('The pool is EMPTY!')
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

    def load_fund_pool(self, date):

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

    def dates_divided(self):

        dates = []
        date = self._begin_date
        while date <= self._end_date:
            dates.append(date)
            date = month_start(next_month(date))
        dates.append(self._end_date + pd.Timedelta('1d'))

        return dates

    @abstractmethod
    def estimate_fund_inc(self):

        pass
        # raise NotImplementedError()

# xx = FundIncEstimate(fund_codes=['000001', '000011'], end_date='20181228')
# xx.estimate_fund_inc()
# set_trace()

class FundIncEstSkPos(FundIncEstimate):

    # __df_stock_ret = None

    def __init__(self, begin_date=None, end_date=None, fund_ids=None, fund_codes=None):

        super(FundIncEstSkPos, self).__init__(begin_date, end_date, fund_ids, fund_codes)

    @staticmethod
    def cal_sk_pos(fund_stock_pos, fund_stock_pos_ten):

        fund_stock_pos_new = pd.DataFrame()
        for fund_id in fund_stock_pos.index.levels[0]:

            stock_pos = fund_stock_pos.loc[fund_id]

            if fund_id in fund_stock_pos_ten.index.levels[0]:

                stock_pos_ten = fund_stock_pos_ten.loc[fund_id]

                min_navrto = stock_pos_ten.navrto.min()
                stock_pos[stock_pos.navrto > min_navrto] = min_navrto
                stock_pos = stock_pos.drop(stock_pos_ten.index.intersection(stock_pos.index))
                stock_pos = pd.concat([stock_pos, stock_pos_ten])

            stock_pos = stock_pos.reset_index()
            stock_pos['fund_id'] = fund_id

            fund_stock_pos_new = pd.concat([fund_stock_pos_new, stock_pos])

        for fund_id in fund_stock_pos_ten.index.levels[0].difference(fund_stock_pos.index.levels[0]):

            stock_pos_ten = fund_stock_pos_ten.loc[fund_id]
            stock_pos_ten = stock_pos_ten.reset_index()
            stock_pos_ten['fund_id'] = fund_id

            fund_stock_pos_new = pd.concat([fund_stock_pos_new, stock_pos_ten])

        fund_stock_pos_new = fund_stock_pos_new.set_index(['fund_id', 'stock_id']).sort_index()

        return fund_stock_pos_new

    def cal_sk_pos_days(self):

        if self._begin_date.month < 4:
            begin_date = pd.Timestamp(self._begin_date.year-1, 9, 1)
        elif self._begin_date.month < 9:
            begin_date = pd.Timestamp(self._begin_date.year, 4, 1)
        else:
            begin_date = pd.Timestamp(self._begin_date.year, 9, 1)
        dates = pd.Series(index=[begin_date, self._end_date])
        dates = dates.resample('MS').first().index

        df_fund_stock_pos = pd.DataFrame()
        for date in dates:

            print('cal_sk_pos_days', date)

            if date.month in [4, 9]:
                lastest_end_date = last_end_date_fund_skdetail_all_published(date)
                fund_stock_pos = caihui_tq_fd_skdetail.load_fd_skdetail_all(
                        lastest_end_date.strftime('%Y%m%d'),
                        fund_ids=self._fund_pool.index
                ).dropna()
            elif date.month in [2, 5, 8, 11]:
                lastest_end_date = last_end_date_fund_skdetail_ten_published(date)
                fund_stock_pos_ten = caihui_tq_fd_skdetail.load_fd_skdetail_ten(
                        lastest_end_date.strftime('%Y%m%d'),
                        date.strftime('%Y%m%d'),
                        fund_ids=self._fund_pool.index
                ).dropna()
                fund_stock_pos = FundIncEstSkPos.cal_sk_pos(fund_stock_pos, fund_stock_pos_ten)

            fund_stock_pos['date'] = date
            df_fund_stock_pos = pd.concat([df_fund_stock_pos, fund_stock_pos])
            fund_stock_pos = fund_stock_pos.drop('date', axis='columns')

        df_fund_stock_pos = df_fund_stock_pos.reset_index().set_index(['date', 'fund_id', 'stock_id']) / 100.0

        return df_fund_stock_pos

    def estimate_fund_inc(self):

        # trade_dates = ATradeDate.trade_date()
        # mask = (trade_dates >= self._begin_date) & (trade_dates <= self._end_date)
        # if FundIncEstSkPos.__df_stock_ret is not None and trade_dates[mask].isin(FundIncEstSkPos.__df_stock_ret.index).all():
            # df_stock_ret = FundIncEstSkPos.__df_stock_ret.loc[self._begin_date:self._end_date]
        # else:
            # df_stock_nav = caihui_tq_sk_dquoteindic.load_stock_nav(
                    # begin_date=trade_date_before(self._begin_date).strftime('%Y%m%d'),
                    # end_date=self._end_date.strftime('%Y%m%d')
            # ).dropna(how='all', axis='columns')
            # df_stock_ret = df_stock_nav.pct_change().iloc[1:]
            # if FundIncEstSkPos.__df_stock_ret is None:
                # FundIncEstSkPos.__df_stock_ret = df_stock_ret
            # else:
                # index_concat = FundIncEstSkPos.__df_stock_ret.index.difference(df_stock_ret.index)
                # FundIncEstSkPos.__df_stock_ret = pd.concat([FundIncEstSkPos.__df_stock_ret.loc[index_concat], df_stock_ret], how='outer').sort_index()

        dates = self.dates_divided()

        df_stock_nav = caihui_tq_sk_dquoteindic.load_stock_nav(
                begin_date=trade_date_before(self._begin_date).strftime('%Y%m%d'),
                end_date=self._end_date.strftime('%Y%m%d')
        ).dropna(how='all', axis='columns')
        df_stock_ret = df_stock_nav.pct_change().iloc[1:]

        df_fund_stock_pos = self.cal_sk_pos_days()

        df_fund_inc_est = pd.DataFrame()
        for date, next_date in zip(dates[:-1], dates[1:]):

            next_date -= pd.Timedelta('1d')

            print('fund_inc_estimate_sk_pos', date, next_date)

            fund_stock_pos = df_fund_stock_pos.loc[month_start(date)]
            fund_stock_pos = fund_stock_pos.unstack(level=0).navrto.fillna(0.0)
            stock_ret = df_stock_ret.loc[date:next_date].reindex(fund_stock_pos.index, axis='columns').fillna(0.0)

            fund_inc_est = pd.DataFrame(
                    np.dot(stock_ret, fund_stock_pos),
                    index=stock_ret.index,
                    columns=fund_stock_pos.columns
            )
            df_fund_inc_est = pd.concat([df_fund_inc_est, fund_inc_est])

        df_fund_inc_est1 = df_fund_inc_est.rename(self._fund_pool.fund_code, axis='columns').sort_index(axis='columns')
        df_fund_inc_est1.columns.name = 'fund_code'

        print('fund_inc_estimate_sk_pos done.')

        return df_fund_inc_est

if 1==2:
    start=time.perf_counter()
    yy= FundIncEstSkPos(fund_codes=['000001', '000011'], begin_date='20171111', end_date='20181228')
    yy.estimate_fund_inc()
    print(time.perf_counter()-start)
    set_trace()


class FundIncEstIxPos(FundIncEstimate):

    # __df_stock_ret = None

    def __init__(self, begin_date=None, end_date=None, fund_ids=None, fund_codes=None, index_ids=None):

        super(FundIncEstIxPos, self).__init__(begin_date, end_date, fund_ids, fund_codes)

        if index_ids is None:
            self._index_ids = caihui_tq_ix_basicinfo.load_index_basic_info(est_class='申万一级行业指数').index
        else:
            self._index_ids = index_ids

    @property
    def index_ids(self):
        return self._index_ids

    @staticmethod
    def fund_inc_objective(x, pars):

        index_inc, fund_inc = pars
        fund_inc = fund_inc.values
        pre_inc = np.dot(index_inc, x)
        loss = np.sum(np.power((fund_inc-pre_inc), 2))

        return loss

    @staticmethod
    def cal_ix_pos(index_inc, fund_inc):

        index_num = index_inc.shape[1]

        w0 = np.array([1.0/index_num]*index_num)

        cons = (
                {'type': 'ineq', 'fun': lambda x: -sum(x) + 1.0},
                {'type': 'ineq', 'fun': lambda x: sum(x)},
        )

        bnds = tuple([(0.0, 1.0) for i in range(index_num)])

        res = minimize(
                FundIncEstIxPos.fund_inc_objective,
                w0,
                args=[index_inc, fund_inc],
                method='SLSQP',
                constraints=cons,
                options={'disp': False},
                bounds=bnds
        )

        return res.x

    def cal_ix_pos_days(self):

        begin_date_cal = month_start(last_quarter(self._begin_date))
        dates = pd.Series(index=[begin_date_cal, self._end_date])
        dates = dates.resample('MS').first().index

        df_fund_nav = base_ra_fund_nav.load_daily(
                trade_date_before(begin_date_cal).strftime('%Y%m%d'),
                self._end_date.strftime('%Y%m%d'),
                codes=self._fund_pool.fund_code
        )
        df_fund_inc = df_fund_nav.pct_change().iloc[1:]

        df_index_nav = caihui_tq_qt_index.load_index_nav(
                begin_date=trade_date_before(begin_date_cal).strftime('%Y%m%d'),
                end_date=self._end_date.strftime('%Y%m%d'),
                index_ids=self._index_ids
        )
        df_index_inc = df_index_nav.pct_change().iloc[1:]

        lookback = 3
        df_fund_index_pos = pd.DataFrame()
        for date, next_date in zip(dates[:-lookback], dates[lookback:]):

            print('cal_ix_pos_days', next_date)

            next_date -= pd.Timedelta('1d')

            fund_inc = df_fund_inc.loc[date:next_date].dropna(how='any', axis='columns')
            index_inc = df_index_inc.loc[date:next_date]

            fund_index_pos = pd.DataFrame(columns=self._index_ids)
            fund_index_pos.index.name = 'fund_code'
            for fund_code in fund_inc.columns:
                fund_index_pos.loc[fund_code] = FundIncEstIxPos.cal_ix_pos(index_inc, fund_inc[fund_code])
            fund_index_pos['date'] = next_date + pd.Timedelta('1d')
            fund_index_pos = fund_index_pos.reset_index().set_index(['date', 'fund_code'])

            df_fund_index_pos = pd.concat([df_fund_index_pos, fund_index_pos])

        return df_fund_index_pos

    def estimate_fund_inc(self):

        dates = self.dates_divided()

        df_index_nav = caihui_tq_qt_index.load_index_nav(
                begin_date=trade_date_before(self._begin_date).strftime('%Y%m%d'),
                end_date=self._end_date.strftime('%Y%m%d'),
                index_ids=self._index_ids
        )
        df_index_inc = df_index_nav.pct_change().iloc[1:]

        df_fund_index_pos = self.cal_ix_pos_days()

        df_fund_inc_est = pd.DataFrame()
        for date, next_date in zip(dates[:-1], dates[1:]):

            next_date -= pd.Timedelta('1d')

            print('fund_inc_estimate_ix_pos', date, next_date)

            fund_index_pos = df_fund_index_pos.loc[month_start(date)].T
            index_inc = df_index_inc.loc[date:next_date]

            fund_inc_est = pd.DataFrame(np.dot(index_inc, fund_index_pos), index=index_inc.index, columns=fund_index_pos.columns)
            df_fund_inc_est = pd.concat([df_fund_inc_est, fund_inc_est])

        print('fund_inc_estimate_ix_pos done.')

        return df_fund_inc_est

if 1==2:
    ww = FundIncEstIxPos(fund_codes=['000001', '000011'], begin_date='20171111', end_date='20181228')
    ww.estimate_fund_inc()
    set_trace()


class FundIncEstMix(FundIncEstimate):

    # __df_stock_ret = None

    def __init__(self, begin_date=None, end_date=None, fund_ids=None, fund_codes=None):

        super(FundIncEstMix, self).__init__(begin_date, end_date, fund_ids, fund_codes)

    def cal_est_pos():

         asset_fund_inc_estimate.load_fund_inc_estimate()

    def estimate_fund_inc(self):

        dates = self.dates_divided()

        df_est = asset_fund_inc_estimate.load_fund_inc_estimate(
                begin_date=self._begin_date.strftime('%Y%m%d'),
                end_date=self._end_date.strftime('%Y%m%d'),
                fund_codes=self._fund_pool.fund_code,
                methods=['sk_pos', 'ix_pos'],
        )



