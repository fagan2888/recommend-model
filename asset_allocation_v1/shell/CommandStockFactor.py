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
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund
from db.asset_stock_factor import *
from db.asset_stock import *
from db import asset_trade_dates
from multiprocessing import Pool
import math
import scipy.stats as stats
import json
from stock_factor import *
import functools


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def sf(ctx):
    '''multi factor
    '''
    pass


@sf.command()
@click.option('--filepath', 'optfilepath', help=u'stock factor infos')
@click.pass_context
def factor_info(ctx, optfilepath):
    '''insert factor info
    '''
    sf_df = pd.read_csv(optfilepath.strip())
    sf_df.factor_formula = sf_df.factor_formula.where((pd.notnull(sf_df.factor_formula)), None)

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    for i in range(0, len(sf_df)):

        record = sf_df.iloc[i]

        factor = stock_factor()
        factor.sf_id = record.sf_id
        factor.sf_name = record.factor_name
        factor.sf_explain = record.factor_explain
        factor.sf_source = record.factor_source
        factor.sf_kind = record.factor_kind
        factor.sf_formula = record.factor_formula
        factor.sf_start_date = record.start_date

        session.merge(factor)

    session.commit()
    session.close()


def call_factor_func(f):
    return f()


@sf.command()
@click.pass_context
def stock_factor_value(ctx):


    #bp_factor()
    factor_funcs = [ln_capital_factor, ln_price_factor, highlow_price_factor, relative_strength_factor, std_factor, trade_volumn_factor, turn_rate_factor, weighted_strength_factor, bp_factor, current_ratio_factor, cash_ratio_factor, pe_ttm_factor, roa_factor, roe_factor, holder_factor, fcfp_factor]
    #factor_funcs = [bp_factor, current_ratio_factor, cash_ratio_factor, pe_ttm_factor, roa_factor, roe_factor, holder_factor, fcfp_factor]
    pool = Pool(16)
    pool.map(call_factor_func, factor_funcs)
    pool.close()
    pool.join()

    #highlow_price_factor()
    #bp_factor()
    #asset_turnover_factor()
    #current_ratio_factor()
    #cash_ratio_factor()
    #pe_ttm_factor()
    #roa_factor()
    #roe_factor()
    #grossprofit_factor()
    #profit_factor()
    #holder_factor()
    #fcfp_factor()

    return

@sf.command()
@click.pass_context
def stock_valid(ctx):

    stock_factor_util.compute_stock_valid()
    return


@sf.command()
@click.pass_context
def stock_factor_layer_rankcorr(ctx):

    sf_ids = ['SF.000001','SF.000002', 'SF.000015', 'SF.000016','SF.000017','SF.000018', 'SF.000035', 'SF.000036', 'SF.000037', 'SF.000038', 
            'SF.000047','SF.000048','SF.000049','SF.000050','SF.000051','SF.000052','SF.000053','SF.000054','SF.000055','SF.000056',
            'SF.000057','SF.000058','SF.000059','SF.000060','SF.000061','SF.000062','SF.000005','SF.000007','SF.000009','SF.000019','SF.000020',
            'SF.000021','SF.000039','SF.000041',
            ]

    pool = Pool(20)
    pool.map(functools.partial(compute_stock_factor_layer_spearman, 10), sf_ids)
    pool.close()
    pool.join()

    #stock_factor_layer_spearman('SF.000041')

    return


@sf.command()
@click.pass_context
def stock_factor_index(ctx):


    sf_ids = ['SF.000001','SF.000002', 'SF.000015', 'SF.000016','SF.000017','SF.000018', 'SF.000035', 'SF.000036', 'SF.000037', 'SF.000038', 
            'SF.000047','SF.000048','SF.000049','SF.000050','SF.000051','SF.000052','SF.000053','SF.000054','SF.000055','SF.000056',
            'SF.000057','SF.000058','SF.000059','SF.000060','SF.000061','SF.000062','SF.000005','SF.000007','SF.000009','SF.000019','SF.000020',
            'SF.000021','SF.000039','SF.000041',
            ]


    pool = Pool(16)
    pool.map(compute_stock_factor_index, sf_ids)
    pool.close()
    pool.join()

    #stock_factor_index('SF.000041')
    return


@sf.command()
@click.pass_context
def rankcorr_multi_factor(ctx):

    compute_rankcorr_multi_factor_pos()
