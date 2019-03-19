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
import scipy
import scipy.optimize


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

    print(asset_risk_df)
    print(asset_liquidity_df)

    user_asset = {'risk_7':10000, 'steady':2000}
    user_best = {'risk': 0.6, 'liquidity': 0.8}
    assets = ['risk_7', 'steady']
    all_amount = np.sum(list(user_asset.values()))


    def user_risk_liquidity(user_asset):

        user_asset_ser = pd.Series(user_asset)
        user_asset_ser = user_asset_ser / user_asset_ser.sum()
        user_risk = 0
        user_liquidity = 0
        for asset in user_asset_ser.index:
            user_risk = user_risk + asset_risk_df.loc[asset] * user_asset_ser.loc[asset]
            user_liquidity = user_liquidity + asset_liquidity_df.loc[asset] * user_asset_ser.loc[asset]
        return user_risk, user_liquidity


    def user_add_asset_obj(x, user_best, user_asset, assets):
        user_add_asset = user_asset.copy()
        for i in range(0, len(assets)):
            asset = assets[i]
            user_add_asset[asset] = user_add_asset[asset] + x[i]
        user_risk, user_liquidity = user_risk_liquidity(user_add_asset)
        return (user_best['risk'] - user_risk) ** 2 + (user_best['liquidity'] - user_liquidity) ** 2

    x = [0, 0]
    bound = [ (0.0 , all_amount) for i in range(len(assets))]
    cons = [{'type': 'ineq', 'fun': lambda x: all_amount - np.sum(x)}]
    result = scipy.optimize.minimize(user_add_asset_obj, x, (user_best, user_asset, assets), method='SLSQP', bounds = bound, constraints = cons, options = {'ftol':1e-12, 'eps':100})
    if result.success:
        print(result.x)



@account.command()
@click.pass_context
def account_balance_simple(ctx):

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

    print(asset_risk_df)
    print(asset_liquidity_df)

    def user_risk_liquidity(user_asset):

        user_asset_ser = pd.Series(user_asset)
        user_asset_ser = user_asset_ser / user_asset_ser.sum()
        user_risk = 0
        user_liquidity = 0
        for asset in user_asset_ser.index:
            user_risk = user_risk + asset_risk_df.loc[asset] * user_asset_ser.loc[asset]
            user_liquidity = user_liquidity + asset_liquidity_df.loc[asset] * user_asset_ser.loc[asset]
        return user_risk, user_liquidity

    user_asset = {'risk_7':10000, 'steady':2000}
    user_risk, user_liquidity = user_risk_liquidity(user_asset)

    age = 20
    amount = np.sum(list(user_asset.values()))
    age_liquidity = 0.5 + (0.33 / 400) * ((age - 40) ** 2)
    amount_liquidity = 0.5
    if amount <= 20000:
        amount_liquidity = 0.83
    elif amount > 20000 and amount <= 200000:
        amount_liquidity = 0.66
    elif amount > 100000 and amount <= 1000000:
        amount_liquidity = 0.5
    else:
        amount_liquidity = 0.33

    user_asset_risk = 0.0
    for asset in user_asset.keys():
        if asset.startswith('risk'):
            user_asset_risk = asset_liquidity_df.loc[asset]

    print(user_asset_risk, age_liquidity, amount_liquidity)
    user_best_liquidity = 0.4 * user_asset_risk + 0.3 * age_liquidity + 0.3 * amount_liquidity
    add_amount = 0.0
    if user_liquidity > user_best_liquidity:
        add_amount = 0.0
    else:
        add_amount =
