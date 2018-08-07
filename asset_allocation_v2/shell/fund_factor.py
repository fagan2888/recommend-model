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
from asset import StockAsset, StockFundAsset
from db import asset_trade_dates
from db.asset_stock_factor import *
from db import asset_fund_factor
import math
import scipy.stats as stats
import json
from asset import Asset,FundAsset,StockFundAsset
from stock_factor import StockFactor
from functools import reduce
from trade_date import ATradeDate
from ipdb import set_trace
from multiprocessing import Pool
#from pathos.multiprocessing import ProcessingPool as Pool


logger = logging.getLogger(__name__)


class FundFactor(Factor):


    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(FundFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)


    def cal_factor_exposure(self):

        sf_ids = ['SF.0000%02d'%i for i in range(1,10)]+['SF.1000%02d'%i for i in range(1,29)]
        ff_ids = ['FF.0000%02d'%i for i in range(1,10)]+['FF.1000%02d'%i for i in range(1,29)]
        ff_sf = dict(zip(ff_ids, sf_ids))
        sfe = StockFactor.load_factor_exposure(ff_sf[self.factor_id])
        fund_pos = StockFundAsset.all_fund_pos()
        fund_ids = fund_pos.keys()

        sfe = sfe.resample('d').fillna(method = 'pad')
        df_factor_exposure= {}
        for fund_id in fund_ids:
            fp = fund_pos[fund_id]
            dates = fp.index.levels[0]
            df_fund_exposure = {}
            for date in dates:
                pos = fp.loc[date]
                if len(np.intersect1d(sfe.columns, pos.index)) == 0:
                    continue
                pose = sfe.loc[date, pos.index]
                pose = pose.fillna(pose.mean())
                funde = np.dot(pose, pos)[0] / pos.values.sum()
                df_fund_exposure[date] = funde
            df_fund_exposure = pd.Series(df_fund_exposure)
            df_factor_exposure[fund_id] = df_fund_exposure
        df_factor_exposure = pd.DataFrame(df_factor_exposure)
        self.exposure = df_factor_exposure

        return df_factor_exposure


    def cal_factor_return(self, ff_ids):

        period = 21

        close = StockFundAsset.all_fund_nav()
        ret = close.pct_change(period).iloc[period:]
        ret = ret[StockFundAsset.all_fund_info().index]

        dates = ret.index
        dates = dates[dates >= '2010-01-01']
        dates = dates[dates <= '2018-06-30']

        df_ret = pd.DataFrame(columns = ff_ids)
        df_sret = pd.DataFrame(columns = StockFundAsset.all_fund_info().index)

        pool = Pool(len(ff_ids))
        fe = pool.map(Factor.load_factor_exposure, ff_ids)
        pool.close()
        pool.join()
        fed = dict(zip(ff_ids, fe))

        for date, next_date in zip(dates[:-period], dates[period:]):

            print('cal_factor_return:', next_date)
            tmp_exposure = {}
            tmp_ret = ret.loc[next_date].sort_index().dropna()
            for ff in ff_ids:
                tmp_exposure[ff] = fed[ff].loc[date]
            tmp_exposure_df = pd.DataFrame(tmp_exposure)
            tmp_exposure_df = tmp_exposure_df.dropna(how = 'all').sort_index()
            tmp_exposure_df = tmp_exposure_df[ff_ids].fillna(0.0)
            joined_funds = tmp_ret.index.intersection(tmp_exposure_df.index)
            tmp_ret = tmp_ret.loc[joined_funds]
            tmp_exposure_df = tmp_exposure_df.loc[joined_funds]
            mod = sm.OLS(tmp_ret, tmp_exposure_df.values, missing = 'drop').fit()

            df_ret.loc[next_date] = mod.params.values
            df_sret.loc[next_date, joined_funds] = tmp_ret.values - np.dot(tmp_exposure_df.values, mod.params)

        return df_ret, df_sret




if __name__ == '__main__':

    ff = FundFactor('FF.000001')
    ff.cal_factor_exposure()

