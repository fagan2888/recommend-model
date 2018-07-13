#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import os
import logging
import re
import statsmodels.api as sm
from scipy.stats import spearmanr

from ipdb import set_trace

from datetime import datetime, timedelta
from db import database, asset_mz_markowitz, asset_mz_markowitz_alloc, asset_mz_markowitz_argv,  asset_mz_markowitz_asset, asset_mz_markowitz_criteria, asset_mz_markowitz_nav, asset_mz_markowitz_pos, asset_mz_markowitz_sharpe, asset_wt_filter_nav
from db import asset_ra_pool, asset_ra_pool_nav, asset_rs_reshape, asset_rs_reshape_nav, asset_rs_reshape_pos
from db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates, base_exchange_rate_index_nav, asset_ra_bl
from db.asset_stock_factor import *
from asset import Asset, WaveletAsset
from trade_date import ATradeDate


class StockFactorTest(object):

    def __init__(self):

        # self.sfe = pd.read_csv('data/factor/stock_factor_exposure.csv', index_col = ['stock_id', 'sf_id', 'trade_date'], parse_dates = ['trade_date'])
        # self.sfsr = pd.read_csv('data/factor/stock_factor_specific_return.csv', index_col = ['stock_id', 'trade_date'], parse_dates = ['trade_date'])
        self.sfr = pd.read_csv('data/factor/stock_factor_return.csv', index_col = ['sf_id', 'trade_date'], parse_dates = ['trade_date'])

    def exposure_continuity_test(self):

        sfe = self.sfe
        # dates = sfe.index.levels[2]
        dates = sfe.loc[('SK.000001', 'SF.000001')].index
        factors = sfe.index.levels[1]

        dates = dates[::21]
        factors = factors[2:3]
        for factor in factors:
            print
            for date, ndate in zip(dates[:-1], dates[1:]):

                sfe1 = sfe.xs(date, level = 2).xs(factor, level = 1)
                sfe2 = sfe.xs(ndate, level = 2).xs(factor, level = 1)

                stocks = sfe1.index.intersection(sfe2.index)
                sfe1 = sfe1.loc[stocks]
                sfe2 = sfe2.loc[stocks]
                res = spearmanr(sfe1, sfe2)
                print factor, date, ndate
                print res[0], res[1]


    def exposure_collinearity_test(self):

        sfe = self.sfe
        # dates = sfe.index.levels[2]
        dates = sfe.loc[('SK.000001', 'SF.000001')].index
        dates = dates[::21]

        for date in dates:

            sfe1 = sfe.xs(date, level = 2).unstack().dropna()
            sfe1.columns = sfe1.columns.levels[1]
            for factor in sfe1.columns:
                x = sfe1.drop(factor, axis = 1)
                y = sfe1[factor]
                mod = sm.OLS(y, x).fit()
                vif = 1 / (1 - mod.rsquared)
                print date, factor, vif


    def valid_factor_test(self):

        sfr = self.sfr
        sfr = sfr.unstack().T
        sfr.index = sfr.index.levels[1]
        sfr = sfr.rolling(63).mean().dropna()
        vf = sfr.idxmax(axis = 1)
        vf.to_csv('data/vf.csv')
        set_trace()


if __name__ == '__main__':

    sft = StockFactorTest()
    # sft.exposure_continuity_test()
    # sft.exposure_collinearity_test()
    sft.valid_factor_test()










