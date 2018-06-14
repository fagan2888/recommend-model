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
from . import Const
from . import DFUtil
from . import DBData
from . import util_numpy as npu
from . import Portfolio as PF
from .TimingWavelet import TimingWt
import multiprocessing
from multiprocessing import Manager

from datetime import datetime, timedelta
from dateutil.parser import parse
from .Const import datapath
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from .db import database, asset_mz_markowitz, asset_mz_markowitz_alloc, asset_mz_markowitz_argv,  asset_mz_markowitz_asset, asset_mz_markowitz_criteria, asset_mz_markowitz_nav, asset_mz_markowitz_pos, asset_mz_markowitz_sharpe, asset_wt_filter_nav
from .db import asset_ra_pool, asset_ra_pool_nav, asset_rs_reshape, asset_rs_reshape_nav, asset_rs_reshape_pos
from .db import base_ra_index, base_ra_index_nav, base_ra_fund, base_ra_fund_nav, base_trade_dates, base_exchange_rate_index_nav, asset_ra_bl
from .util import xdict
from .util.xdebug import dd
from .wavelet import Wavelet


import traceback, code


logger = logging.getLogger(__name__)


class View(object):


    def __init__(self, globalid, asset_id, view_sr = None, name = None, nav_sr = None, confidence = 0.5):

        if view_sr is None:
            self.__view_sr = pd.Series(0, index = [pd.datetime(1900,1,1)])
        else:
            self.__view_sr= view_sr

        self.__globalid = globalid
        self.__asset_id = asset_id
        self.__confidence = confidence


    @property
    def confidence(self):
        return self.__confidence

    @property
    def globalid(self):
        return self.__globalid

    @property
    def asset_id(self):
        return self.__asset_id


    def view(self, day):

        view_sr = self.__view_sr.copy()
        view_sr = view_sr[view_sr.index <= day]
        view_sr = view_sr.sort_index().dropna()
        if len(view_sr) == 0:
            return 0
        else:
            return view_sr.ravel()[-1]

    @staticmethod
    def load_view(bf_id):
        engine = database.connection('asset')
        Session = sessionmaker(bind=engine)
        session = Session()
        sql = session.query(asset_ra_bl.ra_bl_view.bl_date, asset_ra_bl.ra_bl_view.bl_index_id, asset_ra_bl.ra_bl_view.bl_view).filter(asset_ra_bl.ra_bl_view.globalid == 'BL.000001').statement
        view_df = pd.read_sql(sql, session.bind, index_col = ['bl_date', 'bl_index_id'], parse_dates =  ['bl_date'])
        view_df = view_df.unstack()
        view_df.columns = view_df.columns.droplevel(0)
        view_df = view_df.sort_index()

        session.commit()
        session.close()

        return view_df
