#coding=utf-8
'''
Created on: Apr. 4, 2019
Modified on: Apr. 23, 2019
Author: Shixun Su, Boyang Zhou
Contact: sushixun@licaimofang.com
'''

import sys
import logging
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance, log_likelihood, empirical_covariance, ledoit_wolf_shrinkage, ledoit_wolf
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, FactorAnalysis
from ipdb import set_trace
sys.path.append('shell')
from db import caihui_tq_ix_basicinfo, caihui_tq_qt_index
from trade_date import ATradeDate
from util_timestamp import *


logger = logging.getLogger(__name__)


class AssetCovariance:

    def __init__(self, index_ids, reindex, look_back, is_rolling, **kwargs):

        self._index_ids = pd.Index(index_ids, name='index_id')

        self._reindex = reindex
        self._look_back = look_back
        self._reindex_total = ATradeDate.trade_date(
            begin_date=self.reindex[0],
            end_date=self.reindex[-1],
            lookback=self.look_back+1
        ).rename('trade_date')

        self._is_rolling = is_rolling

        self._df_index_nav = caihui_tq_qt_index.load_index_nav(
            self.index_ids,
            begin_date=self.reindex_total[0].strftime('%Y%m%d'),
            end_date=self.reindex_total[-1].strftime('%Y%m%d'),
            is_fillna=False
        )
        self._df_index_inc = self.df_index_nav.fillna(method='pad').pct_change().iloc[1:]
        self._df_index_inc[self.df_index_nav.isna().iloc[1:]] = np.nan
        self._reindex_total = self.df_index_nav.index

        self._df_asset_cov_estimate = None

    @property
    def index_ids(self):

        return self._index_ids

    @property
    def reindex(self):

        return self._reindex

    @property
    def look_back(self):

        return self._look_back

    @property
    def reindex_total(self):

        return self._reindex_total

    @property
    def is_rolling(self):

        return self._is_rolling

    @property
    def df_index_nav(self):

        return self._df_index_nav

    @property
    def df_index_inc(self):

        return self._df_index_inc

    @property
    def df_asset_cov_estimate(self):

        return self._df_asset_cov_estimate

    def estimate_asset_cov_days(self):

        if self.df_asset_cov_estimate is not None:
            return self._df_asset_cov_estimate

        ser_reindex = pd.Series(self.reindex, index=self.reindex)
        df_asset_cov_estimate = pd.DataFrame(
            index=self.reindex,
            columns=pd.MultiIndex.from_product([self.index_ids, self.index_ids])
        )
        df_asset_cov_estimate.loc[:, :] = ser_reindex.apply(self._estimate_asset_cov)

        self._df_asset_cov_estimate = df_asset_cov_estimate

        return df_asset_cov_estimate

    def _estimate_asset_cov(self, trade_date):

        pass

    def _load_index_nav(self, trade_date):

        reindex = self.reindex_total[self.reindex_total<=trade_date]
        if self.is_rolling:
            reindex = reindex[-1-self.look_back:]

        df_index_nav = self.df_index_nav.reindex(index=reindex)

        return df_index_nav

    def _load_index_inc(self, trade_date):

        reindex = self.reindex_total[self.reindex_total<=trade_date]
        if self.is_rolling:
            reindex = reindex[-1-self.look_back:]

        df_index_inc = self.df_index_inc.reindex(index=reindex[1:])

        return df_index_inc

    def covariance_analysis(self):

        self.estimate_asset_cov_days()

        ser_norm = pd.Series(index=self.reindex[:-1])

        for trade_date, next_trade_date in zip(self.reindex[:-1], self.reindex[1:]):

            df_index_cov_estimate = self.df_asset_cov_estimate.loc[trade_date].unstack()
            df_index_cov = self._load_index_inc(next_trade_date).dropna().cov()
            df_index_cov_delta = df_index_cov_estimate - df_index_cov

            ser_norm.loc[trade_date] = np.linalg.norm(df_index_cov_delta, 'fro')
            # ser_norm.loc[trade_date] = np.linalg.norm(df_index_cov_delta, 'nuc')

        return ser_norm


class AssetCovarianceLast(AssetCovariance):

    def __init__(self, index_ids, reindex, look_back, is_rolling, **kwargs):

        super(AssetCovarianceLast, self).__init__(index_ids, reindex, look_back, is_rolling, **kwargs)

    def _estimate_asset_cov(self, trade_date):

        index_ids = self.index_ids

        df_index_inc = self._load_index_inc(trade_date)

        df_index_cov = df_index_inc.dropna().cov()
        ser_index_cov = df_index_cov.stack().rename(trade_date)

        return ser_index_cov


class AssetCovarianceEmpirical(AssetCovariance):

    def __init__(self, index_ids, reindex, look_back, is_rolling, **kwargs):

        super(AssetCovarianceEmpirical, self).__init__(index_ids, reindex, look_back, is_rolling, **kwargs)

    def _estimate_asset_cov(self, trade_date):

        index_ids = self.index_ids

        df_index_inc = self._load_index_inc(trade_date)

        df_index_cov = pd.DataFrame(empirical_covariance(df_index_inc.dropna()), index=index_ids, columns=index_ids)
        ser_index_cov = df_index_cov.stack().rename(trade_date)

        return ser_index_cov


class AssetCovarianceLedoitWolf(AssetCovariance):

    def __init__(self, index_ids, reindex, look_back, is_rolling, **kwargs):

        super(AssetCovarianceLedoitWolf, self).__init__(index_ids, reindex, look_back, is_rolling, **kwargs)

    def _estimate_asset_cov(self, trade_date):

        index_ids = self.index_ids

        df_index_inc = self._load_index_inc(trade_date)

        df_index_cov = pd.DataFrame(ledoit_wolf(df_index_inc.dropna(), assume_centered=False)[0], index=index_ids, columns=index_ids)
        ser_index_cov = df_index_cov.stack().rename(trade_date)

        return ser_index_cov


if __name__ == '__main__':

    index_ids = [
        '2070000060', # 000300
        '2070000187', # 000905
        '2070000076', # HSI
        '2070006545', # S&P 500
        '2070006599' # NASDAQ 100
    ]
    begin_date = '2010-01-01'
    end_date = '2019-02-28'
    look_back = 1000
    is_rolling = False

    trade_dates = ATradeDate.week_trade_date(begin_date=begin_date, end_date=end_date).rename('trade_date')

    dict_cov = {}
    df_norm = pd.DataFrame()

    dict_cov['Last'] = AssetCovarianceLast(index_ids, trade_dates, look_back, is_rolling)
    dict_cov['Last'].estimate_asset_cov_days()
    df_norm['Last'] = dict_cov['Last'].covariance_analysis()

    dict_cov['Empirical'] = AssetCovarianceEmpirical(index_ids, trade_dates, look_back, is_rolling)
    dict_cov['Empirical'].estimate_asset_cov_days()
    df_norm['Empirical'] = dict_cov['Empirical'].covariance_analysis()

    dict_cov['LedoitWolf'] = AssetCovarianceLedoitWolf(index_ids, trade_dates, look_back, is_rolling)
    dict_cov['LedoitWolf'].estimate_asset_cov_days()
    df_norm['LedoitWolf'] = dict_cov['LedoitWolf'].covariance_analysis()

    print(df_norm.describe())
    set_trace()

