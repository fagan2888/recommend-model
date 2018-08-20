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
from asset import Asset,FundAsset,StockFundAsset,IndexAsset
from stock_factor import StockFactor
from functools import reduce
from trade_date import ATradeDate
from ipdb import set_trace
from multiprocessing import Pool
#from pathos.multiprocessing import ProcessingPool as Pool


logger = logging.getLogger(__name__)


class IndexFactor(Factor):


    def __init__(self, factor_id = None, asset_ids = None, exposure = None, factor_name = None):
        super(IndexFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)


    def cal_factor_exposure(self):

        sf_ids = ['SF.0000%02d'%i for i in range(1,10)]+['SF.1000%02d'%i for i in range(1,28)]
        if_ids = ['IF.0000%02d'%i for i in range(1,10)]+['IF.1000%02d'%i for i in range(1,28)]
        if_sf = dict(zip(if_ids, sf_ids))
        sfe = StockFactor.load_factor_exposure(if_sf[self.factor_id])
        index_pos = IndexAsset.all_index_pos()
        index_ids = index_pos.keys()

        sfe = sfe.resample('d').fillna(method = 'pad')
        df_factor_exposure= {}
        for index_id in index_ids:
            ip = index_pos[index_id]
            ip = ip.resample('m').last()
            dates = ip.index
            dates = dates[dates > '2007-01-31']
            dates = dates[dates < datetime.now()]
            df_index_exposure = {}
            for date in dates:
                pos = ip.loc[date]
                if len(np.intersect1d(sfe.columns, pos.index)) == 0:
                    continue
                pose = sfe.loc[date, pos.index]
                pose = pose.fillna(pose.mean())
                indexe = (pose*pos).sum() / pos.values.sum()
                df_index_exposure[date] = indexe
            df_index_exposure = pd.Series(df_index_exposure)
            df_factor_exposure[index_id] = df_index_exposure
        df_factor_exposure = pd.DataFrame(df_factor_exposure)
        self.exposure = df_factor_exposure

        return df_factor_exposure





if __name__ == '__main__':

    indexfactor = IndexFactor('IF.000001')
    indexfactor.cal_factor_exposure()







