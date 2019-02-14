#coding=utf8


import string
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import os
import time
import logging
import re
import Const
import DFUtil
import DBData
import util_numpy as npu
import Portfolio as PF
from TimingWavelet import TimingWt
import multiprocessing
from multiprocessing import Manager

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import base_trade_dates
from util import xdict
from util.xdebug import dd
from wavelet import Wavelet
from ipdb import set_trace


import traceback, code


logger = logging.getLogger(__name__)



class ATradeDate(object):

    __a_trade_date = None

    @staticmethod
    def trade_date(begin_date = None, end_date = None, lookback = None):

        if ATradeDate.__a_trade_date is None:
            ATradeDate.__a_trade_date = base_trade_dates.load_trade_dates().sort_index()
        td_date = ATradeDate.__a_trade_date.copy()
        if begin_date is not None:
            if lookback is not None:
                tmp_td_date = td_date[td_date.index <= begin_date]
                td_date = td_date[td_date.index >= tmp_td_date.index[-1 * lookback]]
            else:
                td_date = td_date[td_date.index >= begin_date]
        if end_date is not None:
            td_date = td_date[td_date.index <= end_date]
        return td_date.index


    @staticmethod
    def week_trade_date(begin_date = None, end_date = None, lookback = None):

        if ATradeDate.__a_trade_date is None:
            ATradeDate.__a_trade_date = base_trade_dates.load_trade_dates().sort_index()
        td_date = ATradeDate.__a_trade_date.copy()
        td_date = td_date[(td_date.td_type & 0x02) > 0]
        if begin_date is not None:
            if lookback is not None:
                tmp_td_date = td_date[td_date.index <= begin_date]
                td_date = td_date[td_date.index >= tmp_td_date.index[-1 * lookback]]
            else:
                td_date = td_date[td_date.index >= begin_date]
        if end_date is not None:
            td_date = td_date[td_date.index <= end_date]
        return td_date.index


    @staticmethod
    def month_trade_date(begin_date = None, end_date = None, lookback = None):

        if ATradeDate.__a_trade_date is None:
            ATradeDate.__a_trade_date = base_trade_dates.load_trade_dates().sort_index()
        td_date = ATradeDate.__a_trade_date.copy()
        yesterday = td_date.iloc[[-2], :]
        td_date = td_date[(td_date.td_type & 0x08) > 0]
        if not yesterday.index in td_date.index:
            td_date = pd.concat([td_date, yesterday])
        if begin_date is not None:
            if lookback is not None:
                tmp_td_date = td_date[td_date.index <= begin_date]
                td_date = td_date[td_date.index >= tmp_td_date.index[-1 * lookback]]
            else:
                td_date = td_date[td_date.index >= begin_date]
        if end_date is not None:
            td_date = td_date[td_date.index <= end_date]
        return td_date.index


if __name__ == '__main__':
    a_trade_date = ATradeDate()
    print(a_trade_date.trade_date())
    a_trade_date = ATradeDate()
    print(a_trade_date.week_trade_date())
    a_trade_date = ATradeDate()
    print(a_trade_date.month_trade_date())
