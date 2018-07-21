#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import sys
sys.path.append('shell')
import numpy as np
import pandas as pd
from ipdb import set_trace

from db import *
from asset import Asset




class ValidFactor():


    def __init__(self):

        self.factor_ids_large = ['MZ.F000%d0'%i for i in range(1, 10)]
        self.factor_ids_small = ['MZ.F100%d0'%i for i in range(1, 10)]
        self.factor_ids_ind = ['1200000%d'%i for i in range(52, 80)]


    def split_filter(self):

        df_nav_factor_large = {}
        for factor_id in self.factor_ids_large:
            factor_nav = Asset.load_nav_series(factor_id)
            df_nav_factor_large[factor_id] = factor_nav

        df_nav_factor_large = pd.DataFrame(df_nav_factor_large)
        df_ret_factor_large = df_nav_factor_large.pct_change().dropna()

        df_nav_factor_small = {}
        for factor_id in self.factor_ids_small:
            factor_nav = Asset.load_nav_series(factor_id)
            df_nav_factor_small[factor_id] = factor_nav

        df_nav_factor_small = pd.DataFrame(df_nav_factor_small)
        df_ret_factor_small = df_nav_factor_small.pct_change().dropna()

        for i in range(9):

            df_large = df_ret_factor_large.iloc[:, i]
            df_small = df_ret_factor_small.iloc[:, i]

            corr = np.corrcoef(df_large, df_small)[1, 0]
            print(self.factor_ids_large[i], self.factor_ids_small[i], corr)


    def continuity_filter(self):

        ffe = self.ffe.reset_index()
        ffe = ffe.set_index('ff_id')
        factor_ids = ffe.index.unique()

        df_corr = {}
        for factor_id in factor_ids:
            print(factor_id)

            tmp_ffe = ffe.loc[factor_id]
            tmp_ffe = tmp_ffe.set_index(['trade_date', 'fund_id'])
            tmp_ffe = tmp_ffe.unstack()
            tmp_ffe.columns = tmp_ffe.columns.levels[1]
            tmp_ffe = tmp_ffe.resample('d').last()
            tmp_ffe = tmp_ffe.fillna(method = 'pad', limit = 126)
            tmp_ffe = tmp_ffe[tmp_ffe.index >= '2010-01-01']

            factor_corr = []
            dates = pd.date_range('2016-01-01', '2018-04-23', freq = '6M')
            for date, ndate in zip(dates[:-1], dates[1:]):

                pffe = tmp_ffe.loc[date].dropna()
                nffe = tmp_ffe.loc[ndate].dropna()
                jfunds = pffe.index.intersection(nffe.index)

                pffe = pffe.loc[jfunds]
                nffe = nffe.loc[jfunds]
                corr = np.corrcoef(pffe, nffe)[1, 0]
                factor_corr.append(corr)
            df_corr[factor_id] = np.mean(factor_corr)
            # df_corr[factor_id] = factor_corr[-2]

        df_corr = pd.Series(df_corr)
        df_corr = df_corr.sort_values(ascending = False)
        set_trace()

        # print(factor_id, date, ndate, corr)

    def exposure_filter(self):

        ffe = self.ffe.reset_index()
        ffe = ffe.set_index('ff_id')
        factor_ids = ffe.index.unique()

        for factor_id in factor_ids:
            print(factor_id)

            tmp_ffe = ffe.loc[factor_id]
            tmp_ffe = tmp_ffe.set_index(['trade_date', 'fund_id'])
            tmp_ffe = tmp_ffe.unstack()
            tmp_ffe.columns = tmp_ffe.columns.levels[1]
            tmp_ffe = tmp_ffe.resample('d').last()
            tmp_ffe = tmp_ffe.fillna(method = 'pad', limit = 126)
            tmp_ffe = tmp_ffe[tmp_ffe.index >= '2010-01-01']
            tmp_ffe = tmp_ffe.fillna(0.0)
            # ffe_mean = np.sort(tmp_ffe.values, 1)[:, -20:].mean(1)
            ffe_mean = np.sort(tmp_ffe.values, 1)[:, :20].mean(1)
            df_res = pd.Series(data = ffe_mean, index = tmp_ffe.index)
            # print(df_res.tail(10))
            print(df_res.mean())


    def pool_corr(self):

        pool_id = '11115001'
        # pool_id = '11110101'
        index_id = '120000001'

        # pool_nav = asset_ra_pool_nav.load_series('11110101', 0, 9)
        pool_nav = asset_ra_pool_nav.load_series(pool_id, 0, 9)
        pool_ret = pool_nav.resample('w').last().pct_change().dropna()

        index_nav = base_ra_index_nav.load_series(index_id)
        index_nav = index_nav.reindex(pool_nav.index)
        index_ret = index_nav.resample('w').last().pct_change().dropna()
        index_ret = index_ret.reindex(pool_ret.index)

        # corr = np.corrcoef(index_ret, pool_ret)[1, 0]
        corr = np.corrcoef(index_nav, pool_nav)[1, 0]
        print(index_id, pool_id, corr)


    def handle(self):

        # self.split_filter()

        # self.ffe = asset_fund_factor.load_fund_factor_exposure()
        # self.continuity_filter()
        # self.exposure_filter()

        self.pool_corr()


if __name__ == '__main__':

    vf = ValidFactor()
    vf.handle()





