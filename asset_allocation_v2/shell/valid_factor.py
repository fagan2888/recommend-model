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
            dates = pd.date_range('2010-01-01', '2018-04-23', freq = '6M')
            for date, ndate in zip(dates[:-1], dates[1:]):

                pffe = tmp_ffe.loc[date].dropna()
                nffe = tmp_ffe.loc[ndate].dropna()
                jfunds = pffe.index.intersection(nffe.index)

                pffe = pffe.loc[jfunds]
                nffe = nffe.loc[jfunds]
                corr = np.corrcoef(pffe, nffe)[1, 0]
                factor_corr.append(corr)
            # df_corr[factor_id] = np.mean(factor_corr)
            df_corr[factor_id] = factor_corr[-1]

        df_corr = pd.Series(df_corr)
        df_corr = df_corr.sort_values(ascending = False)
        set_trace()

        # print(factor_id, date, ndate, corr)


    def handle(self):

        # self.split_filter()

        self.ffe = asset_fund_factor.load_fund_factor_exposure()
        self.continuity_filter()




if __name__ == '__main__':

    vf = ValidFactor()
    vf.handle()
























