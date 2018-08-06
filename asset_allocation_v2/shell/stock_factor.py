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
from sklearn.linear_model import LinearRegression


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from utils import get_today
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
from asset import Asset
from functools import reduce
from trade_date import ATradeDate
from ipdb import set_trace
from multiprocessing import Pool
#from pathos.multiprocessing import ProcessingPool as Pool



logger = logging.getLogger(__name__)


def multiprocess_load_factor_exposure(sf):
    sf.exposure
    return sf


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
        'SF.000009':'GrowthStockFactor',
        'SF.100001':'FarmingStockFactor',
        'SF.100002':'MiningStockFactor',
        'SF.100003':'ChemicalStockFactor',
        'SF.100004':'FerrousStockFactor',
        'SF.100005':'NonFerrousStockFactor',
        'SF.100006':'ElectronicStockFactor',
        'SF.100007':'CTEquipStockFactor',
        'SF.100008':'HouseholdElecStockFactor',
        'SF.100009':'FoodBeverageStockFactor',
        'SF.100010':'TextileStockFactor',
        'SF.100011':'LightIndustryStockFactor',
        'SF.100012':'MedicalStockFactor',
        'SF.100013':'PublicStockFactor',
        'SF.100014':'ComTransStockFactor',
        'SF.100015':'RealEstateStockFactor',
        'SF.100016':'TradingStockFactor',
        'SF.100017':'TourismStockFactor',
        'SF.100018':'BankStockFactor',
        'SF.100019':'FinancialStockFactor',
        'SF.100020':'CompositeStockFactor',
        'SF.100021':'ConstructionStockFactor',
        'SF.100022':'ArchitecturalStockFactor',
        'SF.100023':'ElecEquipStockFactor',
        'SF.100024':'MachineryStockFactor',
        'SF.100025':'MilitaryStockFactor',
        'SF.100026':'ComputerStockFactor',
        'SF.100027':'MedicalStockFactor',
        'SF.100028':'CommunicationStockFactor',
    }

    __valid_stock_filter = None

    @staticmethod
    def subclass(sf_id, sub_class_name):
        subclasses = StockFactor.__subclasses__() + IndustryStockFactor.__subclasses__()
        for subclass in subclasses:
            if sub_class_name == subclass.__name__:
                return subclass(factor_id = sf_id)

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
            if int(self.factor_id[3:]) <= 9:
                stock_exposure_df = StockFactor.normalized(stock_exposure_df)
            factor_exposure.append(stock_exposure_df)

        factor_exposure_df = reduce(lambda x, y: x+y, factor_exposure)/len(factor_exposure)
        # factor_exposure_df = factor_exposure_df.fillna(method = 'pad')
        factor_exposure_df = factor_exposure_df[all_stocks.index]

        self.exposure = factor_exposure_df

        return factor_exposure_df


    def cal_size(self, stock_id):
        stock_quote = StockAsset.get_stock(stock_id).quote
        totmktcap = stock_quote.totmktcap
        return totmktcap


    def cal_factor_return(self, sf_ids):

        period = 21
        sfs = []
        for sf_id in sf_ids:
            sfs.append(StockFactor.subclass(sf_id, StockFactor.stock_factors[sf_id]))

        close = StockAsset.all_stock_nav()
        ret = close.pct_change(period).iloc[period:]
        ret = ret[StockAsset.all_stock_info().index]

        dates = ret.index
        dates = dates[dates >= '2005-01-01']
        dates = dates[dates <= '2018-06-01']

        df_ret = pd.DataFrame(columns = sf_ids)
        df_sret = pd.DataFrame(columns = StockAsset.all_stock_info().index)
        df_rs = pd.DataFrame(columns = ['Rsquare'])

        pool = Pool(len(sfs))
        sfs = pool.map(multiprocess_load_factor_exposure, sfs)
        pool.close()
        pool.join()

        for date, next_date in zip(dates[:-period], dates[period:]):

            tmp_exposure = {}
            tmp_ret = ret.loc[next_date].values
            for sf in sfs:
                tmp_exposure[sf.factor_id] = sf.exposure.loc[date]
                #tmp_exposure[sf.factor_id] = fed[sf.factor_id].loc[date]
            tmp_exposure_df = pd.DataFrame(tmp_exposure)
            tmp_exposure_df = tmp_exposure_df[sf_ids].fillna(0.0)
            tmp_exposure_df = tmp_exposure_df.loc[StockAsset.all_stock_info().index]
            mod = sm.OLS(tmp_ret, tmp_exposure_df.values, missing = 'drop').fit()
            # mod = sm.WLS(tmp_ret, tmp_exposure_df.values, weights = tmp_amount, missing = 'drop').fit()
            # print(mod.summary())

            df_rs.loc[next_date] = mod.rsquared
            df_ret.loc[next_date] = mod.params
            df_sret.loc[next_date] = tmp_ret - np.dot(tmp_exposure_df.values, mod.params)

            df_rs.to_csv('data/factor_rs.csv')

        return df_ret, df_sret


    #插入股票合法性表
    @staticmethod
    def valid_stock_table():

        all_stocks = StockAsset.all_stock_info()
        all_stocks = all_stocks.reset_index()
        all_stocks = all_stocks.set_index(['sk_secode'])

        st_stocks = StockAsset.stock_st()

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

        quotation = quotation.rename(columns = dict(zip(all_stocks.index, all_stocks.globalid)))
        asset_stock_factor.update_valid_stock_table(quotation)


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
        for desc_method in self.desc_methods:
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

        self.exposure = factor_exposure_df

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
            # partial(self.cal_dastd, period = days),
            # partial(self.cal_dastd, period = days*2),
            # partial(self.cal_dastd, period = days*3),
            # partial(self.cal_dastd, period = days*6),
            # partial(self.cal_dastd, period = days*12),
            # partial(self.cal_hilo, period = days),
            # partial(self.cal_hilo, period = days*2),
            # partial(self.cal_hilo, period = days*3),
            # partial(self.cal_hilo, period = days*6),
            # partial(self.cal_hilo, period = days*12),
            # self.cal_btsg,
            self.cal_cmra,
        ]

    # def cal_dastd(self, stock_id, period = 23):
    #     stock_quote = StockAsset.get_stock(stock_id).quote
    #     close = stock_quote.tclose
    #     close = close.replace(0.0, method = 'pad')
    #     ret = close.pct_change()
    #     ret = ret.rolling(period).apply(lambda x: pow(pow(x,2).mean(), 0.5))

    #     return ret

    # def cal_hilo(self, stock_id, period = 23):
    #     stock_quote = StockAsset.get_stock(stock_id).quote
    #     high = stock_quote.thigh
    #     high = high.replace(0.0, method = 'pad')
    #     hi = high.rolling(period).max()

    #     low = stock_quote.tlow
    #     low = low.replace(0.0, method = 'pad')
    #     low = low.rolling(period).min()

    #     hilo = np.log(hi / low)

    #     return hilo


    def cal_btsg(self, stock_id):

        stock_quote = StockAsset.get_stock(stock_id).quote
        close = stock_quote.tcloseaf
        close = close.replace(0.0, method = 'pad')
        ret = close.pct_change()

        sz = Asset.load_nav_series('120000016')
        bret = sz.pct_change()

        ret = ret.resample('m').sum().iloc[:-1]
        bret = bret.resample('m').sum().iloc[:-1]
        common_index = ret.index.intersection(bret.index)
        ret = ret.loc[common_index]
        bret = bret.loc[common_index]

        ser = pd.Series()

        if len(common_index) < 60:
            return ser

        for i in range(60, len(common_index)):

            tmp_dates = common_index[:i+1]
            y = ret.loc[tmp_dates].values
            x = bret.loc[tmp_dates].values.reshape(-1,1)
            x = sm.add_constant(x)
            mod = sm.OLS(y, x).fit()
            beta = mod.params[1]
            sigma = mod.resid.std()
            btsg = pow(beta * sigma, 0.5)

            ser.loc[tmp_dates[-1]] = btsg

        today = get_today()
        ser.loc[today] = np.nan
        ser = ser.resample('d').last().fillna(method = 'pad')

        return ser


    def cal_cmra(self, stock_id):

        stock_quote = StockAsset.get_stock(stock_id).quote
        close = stock_quote.tcloseaf
        close = close.replace(0.0, method = 'pad')
        nav = close / close.iloc[0]

        zt = np.log(nav)
        zt_max = zt.rolling(window = 252 * 5).max()
        zt_min = zt.rolling(window = 252 * 5).min()
        cmra = np.log((1 + zt_max) / (1 + zt_min))
        cmra = cmra.fillna(method = 'pad')
        cmra = cmra.dropna()

        return cmra


class MomStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(MomStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        days = 21
        self.desc_methods = [
            # partial(self.cal_mom, period = days),
            # partial(self.cal_mom, period = days*2),
            # partial(self.cal_mom, period = days*3),
            # partial(self.cal_mom, period = days*6),
            # partial(self.cal_mom, period = days*60),
            partial(self.cal_halpha),
        ]

    def cal_mom(self, stock_id, period = 23):

        stock_quote = StockAsset.get_stock(stock_id).quote
        close = stock_quote.tcloseaf
        close = close.replace(0.0, method = 'pad')
        ret = close.pct_change()
        tr = stock_quote.turnrate
        ret_tr = (ret*tr).rolling(period).sum()
        weight = tr.rolling(period).sum()
        # mom = ret.rolling(period).mean()
        mom = ret_tr / weight

        return mom

    def cal_halpha(self, stock_id):

        stock_quote = StockAsset.get_stock(stock_id).quote
        close = stock_quote.tcloseaf
        close = close.replace(0.0, method = 'pad')
        ret = close.pct_change()

        sz = Asset.load_nav_series('120000016')
        bret = sz.pct_change()

        ret = ret.resample('m').sum().iloc[:-1]
        bret = bret.resample('m').sum().iloc[:-1]
        common_index = ret.index.intersection(bret.index)
        ret = ret.loc[common_index]
        bret = bret.loc[common_index]

        ser = pd.Series()

        if len(common_index) < 60:
            return ser

        for i in range(60, len(common_index)):

            tmp_dates = common_index[i-59: i+1]
            y = ret.loc[tmp_dates].values
            x = bret.loc[tmp_dates].values.reshape(-1,1)
            mod = LinearRegression().fit(x, y)
            ser.loc[tmp_dates[-1]] = mod.intercept_

        today = get_today()
        ser.loc[today] = np.nan
        ser = ser.resample('d').last().fillna(method = 'pad')

        return ser


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


class GrowthStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(GrowthStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        self.desc_methods = [
            self.cal_egro,
        ]

    def cal_egro(self, stock_id):

        def cal_egro_single(x):

            mod = LinearRegression().fit(np.arange(5).reshape(-1,1), x)
            return mod.coef_[0]/np.mean(x)

        stock_fdmt = StockAsset.get_stock(stock_id).fdmt
        stock_quote = StockAsset.get_stock(stock_id).quote
        p = stock_quote.tclose
        pe = stock_fdmt.pettm
        eps = p / pe
        eps = eps[eps.diff() > 0.001]
        eps_y = pd.Series()
        for k,v in eps.groupby(eps.index.strftime('%Y')):
            eps_y.loc[v.index[-1]] = v.values[-1]

        eps_y = eps_y.rolling(5).apply(cal_egro_single)
        today = datetime.now()
        today_idx = pd.tslib.Timestamp(today.year, today.month, today.day)
        eps_y.loc[today_idx] = np.nan
        eps_y = eps_y.resample('d').last().fillna(method = 'pad').dropna()

        return eps_y


class IndustryStockFactor(StockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None, sf_ind = None):
        super(IndustryStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)
        self.sf_ind = sf_ind
        self.desc_methods = [
            self.cal_indexposure,
        ]

    def cal_indexposure(self, stock_id):

        stock_quote = StockAsset.get_stock(stock_id).quote
        stock_info = StockAsset.all_stock_info()
        sf = pd.DataFrame(index = stock_quote.index)
        sf_ind = stock_info.loc[stock_id].sk_swlevel1code

        if sf_ind == self.sf_ind:
            sf_exposure = 1
        else:
            sf_exposure = 0
        sf['exposure'] = sf_exposure

        return sf.exposure


class FarmingStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(FarmingStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '110000')


class MiningStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(MiningStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '210000')


class ChemicalStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(ChemicalStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '220000')


class FerrousStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(FerrousStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '230000')


class NonFerrousStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(NonFerrousStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '240000')


class ElectronicStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(ElectronicStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '270000')

class CTEquipStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(CTEquipStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '280000')

class HouseholdElecStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(HouseholdElecStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '330000')

class FoodBeverageStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(FoodBeverageStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '340000')

class TextileStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(TextileStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '350000')

class LightIndustryStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(LightIndustryStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '360000')

class MedicalStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(MedicalStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '370000')

class PublicStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(PublicStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '410000')

class ComTransStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(ComTransStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '420000')

class RealEstateStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(RealEstateStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '430000')

class TradingStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(TradingStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '450000')

class TourismStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(TourismStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '460000')

class BankStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(BankStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '480000')

class FinancialStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(FinancialStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '490000')

class CompositeStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(CompositeStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '510000')

class ConstructionStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(ConstructionStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '610000')

class ArchitecturalStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(ArchitecturalStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '620000')

class ElecEquipStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(ElecEquipStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '630000')

class MachineryStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(MachineryStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '640000')

class MilitaryStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(MilitaryStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '650000')

class ComputerStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(ComputerStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '710000')

class MediaStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(MediaStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '720000')

class CommunicationStockFactor(IndustryStockFactor):

    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(CommunicationStockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name, '730000')


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
    # print sf.cal_factor_return(['SF.000001', 'SF.000002', 'SF.000003', 'SF.000004','SF.000005', 'SF.000006','SF.000007', 'SF.000008'])
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
    # asset_stock_factor.update_exposure(SizeStockFactor(factor_id = 'SF.000001'))

    # sf = FarmingStockFactor('SF.000009')
    # print sf.cal_factor_exposure()
    # sf = GrowthStockFactor('SF.000009')
    # sf = MomStockFactor('SF.000009')
    sf = GrowthStockFactor('SF.000009')
    # print sf.cal_factor_exposure()
    print(sf.cal_factor_return(['SF.000009']))






