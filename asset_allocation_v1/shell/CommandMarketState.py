# coding=utf-8

import sys
import click
sys.path.append('shell')
import logging
import pandas as pd
import numpy as np
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from ipdb import set_trace

import config
from db import database, asset_trade_dates, base_ra_index_nav
from db.asset_fundamental import *
from calendar import monthrange
from datetime import datetime, timedelta
from asset import Asset
from trade_date import ATradeDate

import warnings
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri


warnings.filterwarnings("ignore")
r = ro.r
numpy2ri.activate()
pandas2ri.activate()


logger = logging.getLogger(__name__)



@click.group(invoke_without_command=True)
@click.pass_context
def market_state(ctx):
    '''
    macro timing
    '''
    if ctx.invoked_subcommand is None:
        ctx.invoke(hmm_state)
    else:
        pass



@market_state.command()
@click.pass_context
def hmm_state(ctx):

    #asset_ids = ['120000001', '120000010','120000013','120000014']
    asset_ids = ['120000001']
    navs = {}
    for asset_id in asset_ids:
        navs[asset_id] = Asset(asset_id).nav()
    navs = pd.DataFrame(navs)
    navs = navs.reindex(ATradeDate.trade_date()).dropna()

    #print navs.tail()
    #r.regime_gmmhmm(navs, target_index = 4, nstate = 3)

    navs.to_csv('tmp/nav.csv')
