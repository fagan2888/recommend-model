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
import util_numpy as npu
import MySQLdb
import config
from ipdb import set_trace


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func, literal_column
from tabulate import tabulate
from db import database, base_exchange_rate_index, base_ra_index, asset_ra_pool_fund, base_ra_fund, asset_ra_pool, asset_on_online_nav, asset_ra_portfolio_nav, asset_on_online_fund, asset_mz_markowitz_nav, base_ra_index_nav, asset_ra_composite_asset_nav, base_exchange_rate_index_nav, base_ra_fund_nav, asset_mz_highlow_pos, asset_ra_pool_nav
from util import xdict
from trade_date import ATradeDate
from asset import Asset
from monetary_fund_filter import MonetaryFundFilter

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def account(ctx):
    '''
        analysis something
    '''
    pass



@account.command()
@click.pass_context
def account_balance(ctx):

    asset_risk = {  'risk_1':0.015,
                    'risk_2':0.03,
                    'risk_3':0.045,
                    'risk_4':0.06,
                    'risk_5':0.075,
                    'risk_6':0.09,
                    'risk_7':0.105,
                    'risk_8':0.120,
                    'risk_9':0.135,
                    'risk_10':0.15,
                    'steady':0.01,
                    'money':0.0,
            }

    asset_liquidity = {'risk_1':3,
                        'risk_2':3,
                        'risk_3':6,
                        'risk_4':9,
                        'risk_5':12,
                        'risk_6':15,
                        'risk_7':18,
                        'risk_8':24,
                        'risk_9':30,
                        'risk_10':36,
                        'steady':3,
                        'money':0,
            }

    asset_risk_df = pd.Series(asset_risk)
    asset_liquidity_df = pd.Series(asset_liquidity)
    asset_liquidity_df = 1.0 - asset_liquidity_df * 1.0 / 36
    asset_risk_df = (asset_risk_df - min(asset_risk_df)) / (max(asset_risk_df) - min(asset_risk_df))
    asset_liquidity_df = (asset_liquidity_df - min(asset_liquidity_df)) / (max(asset_liquidity_df) - min(asset_liquidity_df))
    #print(asset_risk_df)
    #print(asset_liquidity_df)


    user_asset = {'risk_7':10000, 'steady':2000}
    user_best = {'risk': 0.5, 'liquidity': 0.5}

    user_asset_ser = pd.Series(user_asset)
    user_asset_ser = user_asset_ser / user_asset_ser.sum()

    user_risk = 0
    user_liquidity = 0
    for asset in user_asset_ser.index:
        user_risk = user_risk + asset_risk_df.loc[asset] * user_asset_ser.loc[asset]
        user_liquidity = user_liquidity + asset_liquidity_df.loc[asset] * user_asset_ser.loc[asset]

    print(user_risk, user_liquidity)
    print(user_best)

    if user_best['risk'] >= asset_risk_df.loc['steady'] and user_best['liquidity'] < asset_liquidity_df.loc['steady']:
        print('HeHe')
