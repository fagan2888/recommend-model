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


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import distinct
from tabulate import tabulate
import statsmodels.api as sm
import statsmodels
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund
from db.asset_stock_factor import *
from db.asset_stock import *
from db import asset_trade_dates
from multiprocessing import Pool
import math
import scipy.stats as stats
import json


logger = logging.getLogger(__name__)


class Factor(object):

    def __init__(self, factor_id, asset_ids, exposure = None, ret = None, factor_name = None):

        self.__factor_id = factor_id
        self.__asset_ids = asset_ids
        self.__exposure = exposure
        self.__ret = ret
        self.__factor_name = factor_name

    @property
    def factor_id(self):
        return self.__factor_id

    @property
    def asset_ids(self):
        return self.__asset_ids

    @property
    def exposure(self):
        if self.__exposure is None:
            self.__exposure = self.load_factor_exposure(self.factor_id)

        return self.__exposure

    @exposure.setter
    def exposure(self, exposure):
        self.__exposure = exposure


    @property
    def ret(self):
        if self.__ret is None:
            self.__ret = self.cal_factor_return()

        return self.__ret

    def cal_factor_exposure(self):

        return None

    def cal_factor_return(self):

        return None


    @staticmethod
    def load_factor_exposure(factor_id):
        if factor_id[0:2] == 'SF':
            factor_exposure_df = load_stock_factor_exposure(sf_id = factor_id)
            factor_exposure_df = factor_exposure_df.swaplevel(0, 1).loc[factor_id].swaplevel(0,1).unstack()
            factor_exposure_df.columns = factor_exposure_df.columns.droplevel(0)
            print(factor_id ,' load exposure done')
            return factor_exposure_df
        else:
            pass





