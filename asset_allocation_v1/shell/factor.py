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
import stock_util


logger = logging.getLogger(__name__)


class Factor(object):

    def __init__(self, factor_id, asset_ids, exposure = None, factor_name = None):

        self.__factor_id = factor_id
        self.__asset_ids = asset_ids
        self.__exposure = exposure
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
            self.__exposure = self.cal_factor_exposure()
            return self.__exposure
        else:
            self.__exposure

    def cal_factor_exposure(self):

        return None
