#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ipdb import set_trace

import DBData
from CommandMarkowitz import load_nav_series


class ValidFactor(object):

    def __init__(self, factor_ids, start_date, end_date):

        self.factor_ids = factor_ids
        # self.base_ids = ['120000001', '120000002', '120000018']
        self.base_ids = ['120000001', '120000002']

        self.asset_navs, self.asset_incs = self.load_asset_nav(factor_ids, start_date, end_date)
        self.base_navs, self.base_incs = self.load_asset_nav(self.base_ids, start_date, end_date)


    def load_asset_nav(self, factor_ids, start_date, end_date):

        trade_dates = DBData.trade_dates(start_date, end_date)
        asset_navs = {}
        for factor_id in factor_ids:
            asset_navs[factor_id] = load_nav_series(factor_id, reindex = trade_dates)

        df_asset_navs = pd.DataFrame(asset_navs)
        df_asset_incs = df_asset_navs.pct_change().dropna()

        return df_asset_navs, df_asset_incs


    def cal_base_beta(self):

        for asset in self.factor_ids:
            betas = []
            for base in self.base_ids:
                df_asset = self.asset_incs[asset].values
                df_base = self.base_incs[[base]].values
                lr = LinearRegression()
                lr.fit(df_base, df_asset)
                betas.append(lr.coef_[0])

            betas = np.array(betas)
            invalid_betas = betas[(betas >= 0.9) & (betas < 1.1)]
            if len(invalid_betas) > 0:
                print asset, np.round(betas, 5), 0, betas.argmax()
            else:
                print asset, np.round(betas, 5), 1, betas.argmax()
            # print asset, np.round(betas, 2)


    def handle(self):
        self.cal_base_beta()


if __name__ == '__main__':

    factor_ids = ['1200000%d'%i for i in range(49, 77)]
    # start_date = '2004-01-01'
    # start_date = '2010-06-01'
    start_date = '2005-06-01'
    end_date = '2018-01-22'
    vf = ValidFactor(factor_ids, start_date, end_date)
    vf.handle()
