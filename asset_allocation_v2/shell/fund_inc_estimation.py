#coding=utf-8
'''
Created on: Dec. 28, 2018
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import sys
import logging
from sqlalchemy import MetaData, Table, select, func, and_, not_
import click
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta
sys.path.append('shell')
from db import database, asset_fund_inc_estimate
from db import base_ra_fund_nav, caihui_tq_fd_basicinfo, caihui_tq_fd_skdetail, caihui_tq_sk_dquoteindic, caihui_tq_ix_basicinfo, caihui_tq_qt_index
from trade_date import ATradeDate
from util_timestamp import *


logger = logging.getLogger(__name__)


class FundIncEstimation(object):

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

        if self._begin_date > self._end_date:
            warnings.warn('The begin date is later then the end date.', Warning)

        if fund_codes is None and fund_ids is None:
            self._fund_pool = self.__load_fund_pool(self._end_date)
        else:
            self._fund_pool = caihui_tq_fd_basicinfo.load_fund_code_info(fund_codes=fund_codes, fund_ids=fund_ids)

        if self._fund_pool.empty:
            raise NotImplementedError('The pool of funds is empty.')

    @property
    def fund_pool(self):

        return self._fund_pool

    @property
    def begin_date(self):

        return self._begin_date

    @property
    def end_date(self):

        return self._end_date

    def __load_fund_pool(self, date):

        engine = database.connection('caihui')
        metadata = MetaData(bind=engine)
        t = Table('tq_fd_basicinfo', metadata, autoload=True)

        columns = [
                t.c.SECODE.label('fund_id'),
                t.c.FSYMBOL.label('fund_code')
        ]

        s = select(columns).where(and_(
                t.c.FDSTYLE.in_([2, 3, 4]),
                # t.c.ENABLED==3,
                t.c.FOUNDDATE!='19000101',
                t.c.FOUNDDATE<=(date+relativedelta(years=-1)).strftime('%Y%m%d'),
                t.c.ENDDATE=='19000101',
                t.c.FDNATURE.in_(['证券投资基金', 'LOF']),
                t.c.FDMETHOD=='开放式基金',
                t.c.TOTSHARE>0.1,
                not_(t.c.FDNAME.contains('联接')),
                not_(t.c.FDNAME.contains('沪港深')),
                not_(t.c.FDNAME.contains('港股通'))
        ))

        df = pd.read_sql(s, engine, index_col=['fund_id'])

        return df

    def _dates_divided(self):

        dates = []
        date = self._begin_date
        while date <= self._end_date:
            dates.append(date)
            date += relativedelta(months=+1, day=1)
        dates.append(self._end_date+relativedelta(days=+1))
        dates = pd.Index(dates)

        return dates

    def estimate_fund_inc(self):

        raise NotImplementedError('Method \'estimate_fund_inc\' is not defined.')


class FundIncEstSkPos(FundIncEstimation):

    def __init__(self, begin_date=None, end_date=None, fund_ids=None, fund_codes=None):

        super(FundIncEstSkPos, self).__init__(begin_date, end_date, fund_ids, fund_codes)

    def __cal_sk_pos(self, fund_stock_pos, fund_stock_pos_ten):

        fund_stock_pos_new = pd.DataFrame()
        for fund_id in fund_stock_pos.index.levels[0]:

            stock_pos = fund_stock_pos.loc[fund_id]

            if fund_id in fund_stock_pos_ten.index.levels[0]:

                stock_pos_ten = fund_stock_pos_ten.loc[fund_id]

                min_navrto = stock_pos_ten.navrto.min()
                stock_pos.loc[stock_pos.navrto > min_navrto, 'navrto'] = min_navrto
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

    def __cal_sk_pos_days(self):

        if self._begin_date.month < 4:
            begin_date_cal = pd.Timestamp(self._begin_date.year-1, 9, 1)
        elif self._begin_date.month < 9:
            begin_date_cal = pd.Timestamp(self._begin_date.year, 4, 1)
        else:
            begin_date_cal = pd.Timestamp(self._begin_date.year, 9, 1)
        dates = pd.Series(index=[begin_date_cal, self._end_date])
        dates = dates.resample('MS').first().index

        df_fund_stock_pos = pd.DataFrame()
        for date in dates:

            # print('cal_sk_pos_days', date)

            if date.month in [4, 9]:
                last_end_date = last_end_date_fund_skdetail_all_published(date)
                fund_stock_pos = caihui_tq_fd_skdetail.load_fund_skdetail_all(
                        last_end_date.strftime('%Y%m%d'),
                        fund_ids=self._fund_pool.index
                ).fillna(0.0)
            elif date.month in [2, 5, 8, 11]:
                last_end_date = last_end_date_fund_skdetail_ten_published(date)
                fund_stock_pos_ten = caihui_tq_fd_skdetail.load_fund_skdetail_ten(
                        last_end_date.strftime('%Y%m%d'),
                        date.strftime('%Y%m%d'),
                        fund_ids=self._fund_pool.index
                ).fillna(0.0)
                fund_stock_pos = self.__cal_sk_pos(fund_stock_pos, fund_stock_pos_ten)

            fund_stock_pos['date'] = date
            df_fund_stock_pos = pd.concat([df_fund_stock_pos, fund_stock_pos])
            fund_stock_pos = fund_stock_pos.drop('date', axis='columns')

        df_fund_stock_pos = df_fund_stock_pos.reset_index().set_index(['date', 'fund_id', 'stock_id']) / 100.0

        return df_fund_stock_pos

    def estimate_fund_inc(self):

        dates = self._dates_divided()

        df_stock_nav = caihui_tq_sk_dquoteindic.load_stock_nav(
                begin_date=trade_date_before(self._begin_date).strftime('%Y%m%d'),
                end_date=self._end_date.strftime('%Y%m%d')
        ).dropna(how='all', axis='columns')
        df_stock_ret = df_stock_nav.pct_change().iloc[1:]

        df_fund_stock_pos = self.__cal_sk_pos_days()

        df_fund_inc_est = pd.DataFrame()
        for date, next_date in zip(dates[:-1], dates[1:]):

            next_date += relativedelta(days=-1)
            # print('fund_inc_estimate_sk_pos', date, next_date)

            fund_stock_pos = df_fund_stock_pos.loc[date+relativedelta(day=1)]
            fund_stock_pos = fund_stock_pos.unstack(level=0).navrto.fillna(0.0)
            stock_ret = df_stock_ret.loc[date:next_date].reindex(fund_stock_pos.index, axis='columns').fillna(0.0)

            fund_inc_est = pd.DataFrame(
                    np.dot(stock_ret, fund_stock_pos),
                    index=stock_ret.index,
                    columns=fund_stock_pos.columns
            )
            df_fund_inc_est = pd.concat([df_fund_inc_est, fund_inc_est])

        df_fund_inc_est = df_fund_inc_est.rename(self._fund_pool.fund_code, axis='columns').sort_index(axis='columns')
        df_fund_inc_est.columns.name = 'fund_code'

        # print('fund_inc_estimate_sk_pos done.')

        return df_fund_inc_est


class FundIncEstIxPos(FundIncEstimation):

    def __init__(self, begin_date=None, end_date=None, fund_ids=None, fund_codes=None, index_ids=None):

        super(FundIncEstIxPos, self).__init__(begin_date, end_date, fund_ids, fund_codes)

        if index_ids is None:
            self._index_ids = caihui_tq_ix_basicinfo.load_index_basic_info(est_class='申万一级行业指数').index
        else:
            self._index_ids = pd.Index(index_ids)

    @property
    def index_ids(self):

        return self._index_ids

    def __fund_inc_objective(self, x, pars):

        index_inc, fund_inc = pars
        fund_inc = fund_inc.values
        pre_inc = np.dot(index_inc, x)
        loss = np.sum(np.power((fund_inc-pre_inc), 2))

        return loss

    def __cal_ix_pos(self, index_inc, fund_inc):

        index_num = index_inc.shape[1]

        w0 = np.array([1.0/index_num]*index_num)

        cons = (
                {'type': 'ineq', 'fun': lambda x: 1.0-sum(x)},
                {'type': 'ineq', 'fun': lambda x: sum(x)},
        )

        bnds = tuple([(0.0, 1.0) for i in range(index_num)])

        res = minimize(
                self.__fund_inc_objective,
                w0,
                args=[index_inc, fund_inc],
                method='SLSQP',
                constraints=cons,
                options={'disp': False},
                bounds=bnds
        )

        return res.x

    def __cal_ix_pos_days(self):

        begin_date_cal = self._begin_date + relativedelta(months=-3, day=1)
        dates = pd.Series(index=[begin_date_cal, self._end_date])
        dates = dates.resample('MS').first().index

        df_fund_nav = base_ra_fund_nav.load_daily(
                begin_date=trade_date_before(dates[0]).strftime('%Y%m%d'),
                end_date=trade_date_before(dates[-1]).strftime('%Y%m%d'),
                codes=self._fund_pool.fund_code
        )
        df_fund_inc = df_fund_nav.pct_change().iloc[1:]

        df_index_nav = caihui_tq_qt_index.load_index_nav(
                begin_date=trade_date_before(dates[0]).strftime('%Y%m%d'),
                end_date=trade_date_before(dates[-1]).strftime('%Y%m%d'),
                index_ids=self._index_ids
        )
        df_index_inc = df_index_nav.pct_change().iloc[1:]

        lookback = 3
        df_fund_index_pos = pd.DataFrame()
        for date, next_date in zip(dates[:-lookback], dates[lookback:]):

            # print('cal_ix_pos_days', next_date)
            next_date += relativedelta(days=-1)

            fund_inc = df_fund_inc.loc[date:next_date].dropna(how='any', axis='columns')
            index_inc = df_index_inc.loc[date:next_date]

            fund_index_pos = pd.DataFrame(columns=self._index_ids)
            fund_index_pos.index.name = 'fund_code'
            for fund_code in fund_inc.columns:
                fund_index_pos.loc[fund_code] = self.__cal_ix_pos(index_inc, fund_inc[fund_code])
            fund_index_pos['date'] = next_date + relativedelta(days=+1)
            fund_index_pos = fund_index_pos.reset_index().set_index(['date', 'fund_code'])

            df_fund_index_pos = pd.concat([df_fund_index_pos, fund_index_pos])

        return df_fund_index_pos

    def estimate_fund_inc(self):

        dates = self._dates_divided()

        df_index_nav = caihui_tq_qt_index.load_index_nav(
                begin_date=trade_date_before(self._begin_date).strftime('%Y%m%d'),
                end_date=self._end_date.strftime('%Y%m%d'),
                index_ids=self._index_ids
        )
        df_index_inc = df_index_nav.pct_change().iloc[1:]

        df_fund_index_pos = self.__cal_ix_pos_days()

        df_fund_inc_est = pd.DataFrame()
        for date, next_date in zip(dates[:-1], dates[1:]):

            next_date += relativedelta(days=-1)
            # print('fund_inc_estimate_ix_pos', date, next_date)

            fund_index_pos = df_fund_index_pos.loc[date+relativedelta(day=1)].T
            index_inc = df_index_inc.loc[date:next_date]

            fund_inc_est = pd.DataFrame(np.dot(index_inc, fund_index_pos), index=index_inc.index, columns=fund_index_pos.columns)
            df_fund_inc_est = pd.concat([df_fund_inc_est, fund_inc_est])

        # print('fund_inc_estimate_ix_pos done.')

        return df_fund_inc_est


class FundIncEstMix(FundIncEstimation):

    def __init__(self, begin_date=None, end_date=None, fund_ids=None, fund_codes=None):

        super(FundIncEstMix, self).__init__(begin_date, end_date, fund_ids, fund_codes)

    def __cal_mix_pos(self, fund_inc_estimated, fund_inc):

        fund_inc_estimated_error = pd.Series(index=fund_inc_estimated.columns)
        for method in fund_inc_estimated.columns:
            fund_inc_estimated_error.loc[method] = (fund_inc_estimated[method] - fund_inc).abs().mean() ** 2

        mix_pos = 1 - fund_inc_estimated_error / fund_inc_estimated_error.sum()

        return mix_pos

    def __cal_mix_pos_days(self):

        begin_date_cal = self._begin_date + relativedelta(months=-1, day=1)
        dates = pd.Series(index=[begin_date_cal, self._end_date])
        dates = dates.resample('MS').first().index

        df_fund_nav = base_ra_fund_nav.load_daily(
                begin_date=trade_date_before(dates[0]).strftime('%Y%m%d'),
                end_date=trade_date_before(dates[-1]).strftime('%Y%m%d'),
                codes=self._fund_pool.fund_code
        )
        df_fund_inc = df_fund_nav.pct_change().iloc[1:]

        df_fund_inc_estimated = asset_fund_inc_estimate.load_fund_inc_estimate(
                begin_date=dates[0].strftime('%Y%m%d'),
                end_date=trade_date_before(dates[-1]).strftime('%Y%m%d'),
                fund_codes=self._fund_pool.fund_code,
                methods=['sk_pos', 'ix_pos']
        )

        lookback = 1
        df_fund_mix_pos = pd.DataFrame()
        for date, next_date in zip(dates[:-lookback], dates[lookback:]):

            # print('cal_mix_pos_days', next_date)
            next_date += relativedelta(days=-1)

            fund_inc = df_fund_inc.loc[date:next_date]
            fund_inc_estimated = df_fund_inc_estimated.loc[date:next_date]
            fund_inc_estimated = fund_inc_estimated.unstack().stack(level=0, dropna=False).dropna(axis='columns').unstack()

            fund_mix_pos = pd.DataFrame(columns=['sk_pos', 'ix_pos'])
            fund_mix_pos.index.name = 'fund_code'
            for fund_code in fund_inc.columns.intersection(fund_inc_estimated.columns.levels[0]):
                fund_mix_pos.loc[fund_code] = self.__cal_mix_pos(fund_inc_estimated[fund_code], fund_inc[fund_code])
            fund_mix_pos['date'] = next_date + relativedelta(days=+1)
            fund_mix_pos = fund_mix_pos.reset_index().set_index(['date', 'fund_code'])

            df_fund_mix_pos = pd.concat([df_fund_mix_pos, fund_mix_pos])

        return df_fund_mix_pos

    def estimate_fund_inc(self):

        dates = self._dates_divided()

        df_fund_inc_estimated = asset_fund_inc_estimate.load_fund_inc_estimate(
                begin_date=self._begin_date.strftime('%Y%m%d'),
                end_date=self._end_date.strftime('%Y%m%d'),
                fund_codes=self._fund_pool.fund_code,
                methods=['sk_pos', 'ix_pos']
        )

        df_fund_mix_pos = self.__cal_mix_pos_days()

        trade_dates = df_fund_inc_estimated.index.levels[0]

        df_fund_inc_est = pd.DataFrame(columns=self._fund_pool.fund_code)
        df_fund_inc_est.index.name = 'date'
        for date, next_date in zip(dates[:-1], dates[1:]):

            next_date += relativedelta(days=-1)
            # print('fund_inc_estimate_mix', date, next_date)

            fund_mix_pos = df_fund_mix_pos.loc[date+relativedelta(day=1)]

            for edate in trade_dates[(trade_dates>=date) & (trade_dates<=next_date)]:
                df_fund_inc_est.loc[edate] = (df_fund_inc_estimated.loc[edate] * fund_mix_pos).dropna().sum(axis='columns')

        df_fund_inc_est = df_fund_inc_est.dropna(how='all', axis='columns')

        # print('fund_inc_estimate_mix done.')

        return df_fund_inc_est


def cal_fund_inc_estimate_error():

    df_fund_inc_est = asset_fund_inc_estimate.load_fund_inc_estimate()
    df_fiesp = df_fund_inc_est.sk_pos.unstack()
    df_fieip = df_fund_inc_est.ix_pos.unstack()
    df_fiem = df_fund_inc_est.mix.unstack()

    df_fn = base_ra_fund_nav.load_daily(
            begin_date=trade_date_before(df_fiem.index[0]).strftime('%Y%m%d'),
            end_date=df_fiem.index[-1].strftime('%Y%m%d'),
            codes=df_fiesp.columns
    )
    df_fi = df_fn.pct_change().iloc[1:]

    df_fiesp_error = df_fiesp - df_fi
    df_fieip_error = df_fieip - df_fi
    df_fiem_error = df_fiem - df_fi

    set_trace()


if __name__ == '__main__':

    fie = FundIncEstimation()

    # fiesp = FundIncEstSkPos()
    # df_fiesp = fiesp.estimate_fund_inc()

    # fieip = FundIncEstIxPos(fund_codes=['000001', '000011'])
    # df_fieip = fieip.estimate_fund_inc()

    # fiem = FundIncEstMix(begin_date='20190101', end_date='20190106')
    # df_fiem = fiem.estimate_fund_inc()

    # cal_fund_inc_estimate_error()

