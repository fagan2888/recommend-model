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
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund, asset_fund_factor, asset_index_factor
from db.asset_stock_factor import *
from db.asset_stock import *
from stock_factor import *
from index_factor import IndexFactor
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool


logger = logging.getLogger(__name__)


def exposure_update(indexfactor):
    print(indexfactor.factor_id, 'exposure updating')
    indexfactor.cal_factor_exposure()
    asset_index_factor.update_exposure(indexfactor)


@click.group(invoke_without_command=True)
@click.pass_context
def indexfactor(ctx):
    '''cal index factor
    '''
    pass


@indexfactor.command()
@click.pass_context
def factor_exposure_update(ctx):
    '''insert factor info
    '''

    if_ids = ['IF.0000%02d'%i for i in range(1,10)]+['IF.1000%02d'%i for i in range(1,28)]
    # ff_ids = ['FF.000007']
    ifs = [IndexFactor(if_id) for if_id in if_ids]
    # ffs = [FundFactor('FF.0000%02d'%i) for i in range(1,10)]+[FundFactor('FF.1000%02d'%i) for i in range(1,29)]
    # pool = Pool(len(ffs))
    # pool.map(exposure_update, ffs)
    # pool.close()
    # pool.join()
    for if_ in ifs:
        exposure_update(if_)



