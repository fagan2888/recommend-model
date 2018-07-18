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
from ipdb import set_trace

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund, asset_fund_factor
from db.asset_stock_factor import *
from db.asset_stock import *
from stock_factor import *
from fund_factor import FundFactor
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool


logger = logging.getLogger(__name__)


def exposure_update(ff):
    print(ff.factor_id)
    ff.cal_factor_exposure()
    asset_fund_factor.update_exposure(ff)


@click.group(invoke_without_command=True)
@click.pass_context
def ff(ctx):
    '''multi factor
    '''
    pass


@ff.command()
@click.pass_context
def factor_exposure_update(ctx):
    '''insert factor info
    '''

    ffs = [FundFactor('FF.0000%02d'%i) for i in range(1,10)]+[FundFactor('FF.1000%02d'%i) for i in range(1,29)]

    pool = Pool(len(ffs))
    pool.map(exposure_update, ffs)
    pool.close()
    pool.join()


@ff.command()
@click.pass_context
def factor_return_update(ctx):
    '''insert factor info
    '''

    ffs1 = ['FF.0000%02d'%i for i in range(1, 10)]
    ffs2  = [
        'FF.100004',
        'FF.100008',
        'FF.100009',
        'FF.100012',
        'FF.100018',
        'FF.100019',
        'FF.100025',
        'FF.100026',
    ]
    ffs = ffs1 + ffs2
    ff = FundFactor()
    df_ret, df_sret = ff.cal_factor_return(ffs)

    asset_fund_factor.update_fund_factor_return(df_ret)
    asset_fund_factor.update_fund_factor_specific_return(df_sret)



