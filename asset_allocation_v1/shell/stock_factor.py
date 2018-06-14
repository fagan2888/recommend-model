#coding=utf8


import getopt
import string
import json
import os
import sys
import logging
sys.path.append('shell')
import click
import config
import pandas as pd
import numpy as np
import time
from functools import partial
import statsmodels.api as sm


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import distinct
from tabulate import tabulate
import statsmodels.api as sm
import statsmodels
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund, asset_stock, asset_stock_factor
from factor import Factor
from asset import StockAsset
from db import asset_trade_dates
from db.asset_stock_factor import *
import math
import scipy.stats as stats
import json
import stock_util
from ipdb import set_trace
#from pathos.multiprocessing import ProcessingPool as Pool


logger = logging.getLogger(__name__)


class StockFactor(Factor):

    stock_factors = {
            'SF.000001':'SizeStockFactor',
            'SF.000002':'VolStockFactor',
            'SF.000003':'MomStockFactor',
            'SF.000004':'TurnoverStockFactor',
            'SF.000005':'EarningStockFactor',
            'SF.000006':'ValueStockFactor',
            'SF.000007':'FqStockFactor',
            'SF.000008':'LeverageStockFactor',
            'SF.000009':'SizeNlStockFactor',
            }

    __valid_stock_filter = None

    @staticmethod
    def subclass(sub_class_name):
        for subclass in StockFactor.__subclasses__():
            if sub_class_name == subclass.__name__:
                return subclass()

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(StockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        self.desc_methods = [self.cal_size]

    def cal_factor_exposure(self):
        all_stocks = StockAsset.all_stock_info()
        factor_exposure = []
        for desc_method in self.desc_methods:
            stock_exposure = {}
            for stock_id in all_stocks.index:
                stock_exposure[stock_id] = desc_method(stock_id)
            stock_exposure_df = pd.DataFrame(stock_exposure)
            stock_exposure_df = StockFactor.stock_factor_filter(stock_exposure_df)
            stock_exposure_df = StockFactor.normalized(stock_exposure_df)
            factor_exposure.append(stock_exposure_df)

        factor_exposure_df = reduce(lambda x, y: x+y, factor_exposure)/len(factor_exposure)
        factor_exposure_df = factor_exposure_df.fillna(method = 'pad')
        factor_exposure_df = factor_exposure_df[all_stocks.index]

        self.exposure = factor_exposure_df

        return factor_exposure_df

    def cal_size(self, stock_id):
        stock_quote = StockAsset.get_stock(stock_id).quote
        totmktcap = stock_quote.totmktcap
        return totmktcap

    def cal_factor_return(self, sf_ids):

        sfs = []
        for sf_id in sf_ids:
            sfs.append(StockFactor.subclass(StockFactor.stock_factors[sf_id]))

        close = StockAsset.all_stock_nav()
        ret = close.pct_change()
        ret = ret[StockAsset.all_stock_info().index]

        dates = ret.index
        dates = dates[dates > '2017-01-01']

        df_ret = pd.DataFrame(columns = sf_ids)
        df_sret = pd.DataFrame(columns = StockAsset.all_stock_info().index)
        for date in dates:

            print 'cal_factor_return:', date

            tmp_exposure = []
            tmp_ret = ret.loc[date].values
            for sf in sfs:
                tmp_exposure.append(sf.exposure.fillna(method='pad').loc[date].values.tolist())
            tmp_exposure = np.nan_to_num(np.array(tmp_exposure).T, 0.0)
            mod = sm.OLS(tmp_ret, tmp_exposure, missing = 'drop').fit()

            df_ret.loc[date] = mod.params
            df_sret.loc[date] = tmp_ret - np.dot(tmp_exposure, mod.params)

        return df_ret, df_sret


    #插入股票合法性表
    @staticmethod
    def valid_stock_table():

	all_stocks = StockFactor.get_stock_info()
	all_stocks = all_stocks.reset_index()
	all_stocks = all_stocks.set_index(['sk_secode'])

	st_stocks = stock_util.stock_st()

	all_stocks.sk_listdate = all_stocks.sk_listdate + timedelta(365)

	engine = database.connection('caihui')
	Session = sessionmaker(bind=engine)
	session = Session()
	sql = session.query(asset_stock.tq_qt_skdailyprice.tradedate, asset_stock.tq_qt_skdailyprice.secode ,asset_stock.tq_qt_skdailyprice.tclose, asset_stock.tq_qt_skdailyprice.amount).filter(asset_stock.tq_qt_skdailyprice.secode.in_(all_stocks.index)).statement

	#过滤停牌股票
	quotation_amount = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])

	quotation = quotation_amount[['tclose']]
	quotation = quotation.replace(0.0, np.nan)
	quotation = quotation.unstack()
	quotation.columns = quotation.columns.droplevel(0)

	#60个交易日内需要有25个交易日未停牌
	quotation_count = quotation.rolling(60).count()
	quotation[quotation_count < 25] = np.nan

	#过滤掉过去一年日均成交额排名后20%的股票
	amount = quotation_amount[['amount']]
	amount = amount.unstack()
	amount.columns = amount.columns.droplevel(0)

	year_amount = amount.rolling(252, min_periods = 100).mean()

	def percentile20nan(x):
            x[x <= np.percentile(x,20)] = np.nan
            return x

	year_amount = year_amount.apply(percentile20nan, axis = 1)

	quotation[year_amount.isnull()] = np.nan

	session.commit()
	session.close()

	#过滤st股票
	for i in range(0, len(st_stocks)):
            secode = st_stocks.index[i]
            record = st_stocks.iloc[i]
            selecteddate = record.selecteddate
            outdate = record.outdate
            if secode in set(quotation.columns):
                #print secode, selecteddate, outdate
		quotation.loc[selecteddate:outdate, secode] = np.nan

	#过滤上市未满一年股票
	for secode in all_stocks.index:
            if secode in set(quotation.columns):
		quotation.loc[:all_stocks.loc[secode, 'sk_listdate'], secode] = np.nan


	engine = database.connection('asset')
	Session = sessionmaker(bind=engine)
	session = Session()

	records = session.query(distinct(asset_stock_factor.valid_stock_factor.trade_date)).all()
	dates = [record[0] for record in records]

	dates.sort()
	last_date = dates[-20]

	quotation = quotation[quotation.index >= last_date.strftime('%Y-%m-%d')]

	for date in quotation.index:
            #records = []
            for secode in quotation.columns:
		globalid = all_stocks.loc[secode, 'globalid']
		value = quotation.loc[date, secode]
		if np.isnan(value):
                    continue
		valid_stock = asset_stock_factor.valid_stock_factor()
		valid_stock.stock_id = globalid
		valid_stock.secode = secode
		valid_stock.trade_date = date
		valid_stock.valid = 1.0
		#records.append(valid_stock)
		session.merge(valid_stock)


            #session.add_all(records)
            session.commit()

            logger.info('stock validation date %s done' % date.strftime('%Y-%m-%d'))

	session.commit()
	session.close()

	pass


    #过滤掉不合法股票
    @staticmethod
    def stock_factor_filter(factor_df):

        if StockFactor.__valid_stock_filter is None:

            engine = database.connection('asset')
            Session = sessionmaker(bind=engine)
            session = Session()
            sql = session.query(valid_stock_factor.trade_date,valid_stock_factor.stock_id,  valid_stock_factor.valid).statement
            valid_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'stock_id'], parse_dates = ['trade_date'])
            valid_df = valid_df.unstack()
            valid_df.columns = valid_df.columns.droplevel(0)
            StockFactor.__valid_stock_filter = valid_df
            session.commit()
            session.close()

        if len(factor_df) == 0:
            return factor_df
        else:
            valid_df = StockFactor.__valid_stock_filter.copy()
            valid_df = valid_df.reindex(factor_df.index)
            valid_df = valid_df.reindex_axis(factor_df.columns, axis = 1)
            factor_df[~(valid_df == 1)] = np.nan


	logger.info('vailid filter done')

	return factor_df


    #去极值标准化
    @staticmethod
    def normalized(factor_df):
        factor_df = factor_df.fillna(np.nan)

	#去极值
	factor_median = factor_df.median(axis = 1)

	factor_df_sub_median = abs(factor_df.sub(factor_median, axis = 0))
	factor_df_sub_median_median = factor_df_sub_median.median(axis = 1)

	max_factor_df = factor_median + 10.000 * factor_df_sub_median_median
	min_factor_df = factor_median - 10.000 * factor_df_sub_median_median

        stock_num = len(factor_df.columns)
        stock_ids = factor_df.columns
        max_factor_df = pd.concat([max_factor_df]*stock_num, 1)
        min_factor_df = pd.concat([min_factor_df]*stock_num, 1)
        max_factor_df.columns = stock_ids
        min_factor_df.columns = stock_ids

        factor_df = factor_df.mask(factor_df < min_factor_df, min_factor_df)
        factor_df = factor_df.mask(factor_df > max_factor_df, max_factor_df)

	#归一化
	factor_std  = factor_df.std(axis = 1)
	factor_mean  = factor_df.mean(axis = 1)

	factor_df = factor_df.sub(factor_mean, axis = 0)
	factor_df = factor_df.div(factor_std, axis = 0)

	return factor_df


class SizeStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(SizeStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        self.desc_methods = [self.cal_size]

    def cal_size(self, stock_id):
        stock_quote = StockAsset.get_stock(stock_id).quote
        totmktcap = stock_quote.totmktcap
        return totmktcap


class SizeNlStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(SizeNlStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        self.desc_methods = [self.cal_size]

    def cal_factor_exposure(self):
        all_stocks = StockAsset.all_stock_info()
        factor_exposure = []
        for desc_method in self.__desc_methods:
            stock_exposure = {}
            for stock_id in all_stocks.index:
                stock_exposure[stock_id] = desc_method(stock_id)
            stock_exposure_df = pd.DataFrame(stock_exposure)
            stock_exposure_df = StockFactor.stock_factor_filter(stock_exposure_df)
            stock_exposure_df = StockFactor.normalized(stock_exposure_df)
            stock_exposure_df = pow(stock_exposure_df, 3)
            factor_exposure.append(stock_exposure_df)

        factor_exposure_df = reduce(lambda x, y: x+y, factor_exposure)/len(factor_exposure)
        factor_exposure_df = factor_exposure_df.fillna(method = 'pad')
        factor_exposure_df = factor_exposure_df[all_stocks.index]
        return factor_exposure_df


    def cal_size(self, stock_id):
        stock_quote = StockAsset.get_stock(stock_id).quote
        totmktcap = stock_quote.totmktcap
        return totmktcap


class VolStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(VolStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        days = 23
        self.desc_methods = [
            partial(self.cal_dastd, period = days),
            partial(self.cal_dastd, period = days*2),
            partial(self.cal_dastd, period = days*3),
            partial(self.cal_dastd, period = days*6),
            partial(self.cal_dastd, period = days*12),
            partial(self.cal_hilo, period = days),
            partial(self.cal_hilo, period = days*2),
            partial(self.cal_hilo, period = days*3),
            partial(self.cal_hilo, period = days*6),
            partial(self.cal_hilo, period = days*12),
        ]

    def cal_dastd(self, stock_id, period = 23):
        stock_quote = StockAsset.get_stock(stock_id).quote
        close = stock_quote.tclose
        close = close.replace(0.0, method = 'pad')
        ret = close.pct_change()
        ret = ret.rolling(period).apply(lambda x: pow(pow(x,2).mean(), 0.5))

        return ret

    def cal_hilo(self, stock_id, period = 23):
        stock_quote = StockAsset.get_stock(stock_id).quote
        high = stock_quote.thigh
        high = high.replace(0.0, method = 'pad')
        hi = high.rolling(period).max()

        low = stock_quote.tlow
        low = low.replace(0.0, method = 'pad')
        lo = low.rolling(period).min()

        hilo = np.log(hi / low)

        return hilo


class MomStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(MomStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        days = 23
        self.desc_methods = [
            partial(self.cal_mom, period = days),
            partial(self.cal_mom, period = days*2),
            partial(self.cal_mom, period = days*3),
            partial(self.cal_mom, period = days*6),
            partial(self.cal_mom, period = days*12),
        ]

    def cal_mom(self, stock_id, period = 23):
        stock_quote = StockAsset.get_stock(stock_id).quote
        close = stock_quote.tclose
        close = close.replace(0.0, method = 'pad')
        ret = close.pct_change()
        tr = stock_quote.turnrate
        ret_tr = (ret*tr).rolling(period).sum()
        weight = tr.rolling(period).sum()
        # mom = ret.rolling(period).mean()
        mom = ret_tr / weight

        return mom


class TurnoverStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(TurnoverStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        days = 23
        self.desc_methods = [
            partial(self.cal_turnover, period = days),
            partial(self.cal_turnover, period = days*2),
            partial(self.cal_turnover, period = days*3),
            partial(self.cal_turnover, period = days*6),
            partial(self.cal_turnover, period = days*12),
        ]

    def cal_turnover(self, stock_id, period = 23):
        stock_quote = StockAsset.get_stock(stock_id).quote
        tr = stock_quote.turnrate
        tr = tr.rolling(period).mean()

        return tr


class EarningStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(EarningStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        self.desc_methods = [
            self.cal_earning
        ]

    def cal_earning(self, stock_id):
        stock_fdmt = StockAsset.get_stock(stock_id).fdmt
        stock_quote = StockAsset.get_stock(stock_id).quote
        p = stock_quote.tclose
        pe = stock_fdmt.pettm
        eps = p / pe

        return eps


class ValueStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(ValueStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        self.desc_methods = [
            self.cal_ep,
            self.cal_bp,
        ]


    def cal_ep(self, stock_id):
        stock_fdmt = StockAsset.get_stock(stock_id).fdmt
        ep = 1 / stock_fdmt.pettm

        return ep


    def cal_bp(self, stock_id):
        stock_fdmt = StockAsset.get_stock(stock_id).fdmt
        bp = 1 / stock_fdmt.pb

        return bp


class FqStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(FqStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        self.desc_methods = [
            self.cal_roa,
            self.cal_roe,
            self.cal_sgpmargin,
        ]


    def cal_roa(self, stock_id):
        stock_fdmt = StockAsset.get_stock(stock_id).fdmt
        roa = stock_fdmt.roa

        return roa


    def cal_roe(self, stock_id):
        stock_fdmt = StockAsset.get_stock(stock_id).fdmt
        roe = stock_fdmt.roedilutedcut

        return roe


    def cal_sgpmargin(self, stock_id):
        stock_fdmt = StockAsset.get_stock(stock_id).fdmt
        sgpmargin = stock_fdmt.sgpmargin

        return sgpmargin


class LeverageStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(LeverageStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        self.desc_methods = [
            self.cal_currentrt,
            self.cal_cashrt,
            self.cal_ltmliabtota,
            self.cal_equtotliab,
        ]

    def cal_currentrt(self, stock_id):
        stock_fdmt = StockAsset.get_stock(stock_id).fdmt
        currentrt = stock_fdmt.currentrt

        return currentrt


    def cal_cashrt(self, stock_id):
        stock_fdmt = StockAsset.get_stock(stock_id).fdmt
        cashrt = stock_fdmt.cashrt

        return cashrt


    def cal_ltmliabtota(self, stock_id):
        stock_fdmt = StockAsset.get_stock(stock_id).fdmt
        ltmliabtota = stock_fdmt.ltmliabtota

        return ltmliabtota

    def cal_equtotliab(self, stock_id):
        stock_fdmt = StockAsset.get_stock(stock_id).fdmt
        equtotliab = stock_fdmt.equtotliab

        return equtotliab


if __name__ == '__main__':

    #StockFactor.valid_stock_table()
    # ssf = SizeStockFactor()
    # snsf = SizeNlStockFactor()
    # print ssf.exposure
    # print snsf.exposure

    # sf = MomStockFactor()
    # sf = VolStockFactor()
    # sf = TurnoverStockFactor()
    # sf = ValueStockFactor()
    # sf = FqStockFactor()
    # sf = LeverageStockFactor()
    # print sf.exposure



    # sf = SizeStockFactor()
    # sf = ValueStockFactor()
    #StockAsset.all_stock_nav()
    #StockAsset.all_stock_quote()
    #for stock_id in StockAsset.all_stock_info().index:
    #    print StockAsset.get_stock(stock_id).quote.tail()
    #StockAsset.all_stock_fdmt()
    #sf = StockFactor()
    #print sf.cal_factor_return(['SF.000001', 'SF.000002', 'SF.000003', 'SF.000004','SF.000005', 'SF.000006','SF.000007', 'SF.000008'])
    #sf = SizeStockFactor()
    #print sf.exposure.tail()
    #sf = VolStockFactor()
    #print sf.exposure.tail()
    # set_trace()
    #print StockFactor.subclass('SizeStockFactor')
    #print sys.modules[__name__]
    #a = getattr(sys.modules[__name__], 'StockFactor')()
    #print a.exposure

    #print StockFactor.__subclasses__()

    #StockFactor.stock_factor_filter(pd.DataFrame())
    #StockFactor.stock_factor_filter(pd.DataFrame())
    #StockFactor.stock_factor_filter(pd.DataFrame())
    asset_stock_factor.update_exposure(SizeStockFactor(factor_id = 'SF.000001'))
