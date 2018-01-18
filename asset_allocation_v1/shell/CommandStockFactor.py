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

    #for i in range(0, len(sf_ids)):
    #    sf_id = sf_ids[i]
    #    print sf_id
    #compute_stock_factor_layer_spearman(5, 'SF.000005')
    pool = Pool(20)
    pool.map(functools.partial(compute_stock_factor_layer_spearman, 5), sf_ids)
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
def select_stock_factor_layer(ctx):

    all_stocks = all_stock_info()
    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index('globalid')

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    factor_type = session.query(stock_factor.sf_id, stock_factor.sf_kind).all()
    factor_type_dict = {}
    for record in factor_type:
        factor_type_dict[record[0]] = record[1]
    #print factor_type_dict

    sql = session.query(stock_factor_rankcorr.sf_id, stock_factor_rankcorr.trade_date, stock_factor_rankcorr.rankcorr).statement
    rankcorr_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'sf_id'], parse_dates = ['trade_date'])

    rankcorr_df = rankcorr_df.unstack()
    rankcorr_df.columns = rankcorr_df.columns.droplevel(0)

    rankcorr_df = rankcorr_df.rolling(14).mean().iloc[14:,]
    rankcorr_abs_df = abs(rankcorr_df)

    stock_pos = {}
    factor_pos = {}
    for date in rankcorr_abs_df.index:
        rankcorr_abs = rankcorr_abs_df.loc[date]
        rankcorr_abs = rankcorr_abs.sort_values(ascending = False)

        date_stocks = stock_pos.setdefault(date, [])
        date_factors = factor_pos.setdefault(date, [])

        has_type = set()
        for index in rankcorr_abs.index:
            rankcorr = rankcorr_df.loc[date, index]
            layer = None
            if rankcorr >= 0:
                layer = 0
            else:
                layer = 4

            record = session.query(stock_factor_layer.stock_ids).filter(and_(stock_factor_layer.sf_id == index,stock_factor_layer.trade_date == date,
                                    stock_factor_layer.layer == layer)).first()

            if record is None:
                continue

            factor_type = factor_type_dict[index]
            if factor_type in has_type:
                continue
            else:
                has_type.add(factor_type)

            date_stocks.extend(json.loads(record[0]))
            date_factors.append([index, layer])

            if len(date_factors) >= 5:
                break

    factor_pos_df = pd.DataFrame(factor_pos).T
    #print factor_pos_df


    #计算每期股票的仓位
    globalid_secode_dict = dict(zip(all_stocks.index.ravel(), all_stocks.sk_secode.ravel()))
    dates = list(stock_pos.keys())
    dates.sort()
    stock_pos_df = pd.DataFrame(0, index = dates, columns = all_stocks.index)
    for date in dates:
        stocks = stock_pos[date]
        stocks = list(set(stocks))
        record = stock_pos_df.loc[date]
        record[record.index.isin(stocks)] = 1.0 / len(stocks)
        stock_pos_df.loc[date] = record
    stock_pos_df = stock_pos_df.rename(columns = globalid_secode_dict)

    session.commit()
    session.close()


    caihui_engine = database.connection('caihui')
    caihui_Session = sessionmaker(bind=caihui_engine)
    caihui_session = caihui_Session()

    sql = caihui_session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.secode ,tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.tradedate >= stock_pos_df.index[0].strftime('%Y%m%d')).statement
    stock_yield_df = pd.read_sql(sql, caihui_session.bind, index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']) / 100.0
    stock_yield_df = stock_yield_df.unstack()
    stock_yield_df.columns = stock_yield_df.columns.droplevel(0)
    secodes = list(set(stock_pos_df.columns) & set(stock_yield_df.columns))
    stock_pos_df = stock_pos_df[secodes]
    stock_yield_df = stock_yield_df[secodes]

    caihui_session.commit()
    caihui_session.close()

    stock_pos_df = stock_pos_df.reindex(stock_yield_df.index).fillna(method = 'pad')
    stock_pos_df = stock_pos_df.shift(1).fillna(0.0)

    factor_yield_df = stock_pos_df * stock_yield_df
    factor_yield_df = factor_yield_df.sum(axis = 1)
    factor_yield_df = factor_yield_df[factor_yield_df.index >= '2006-06-01']
    #print factor_yield_df.index
    factor_nav_df = (1 + factor_yield_df).cumprod()

    factor_nav_df = factor_nav_df.to_frame()
    factor_nav_df.index.name = 'date'
    factor_nav_df.columns = ['nav']
    #print factor_nav_df
    factor_nav_df.to_csv('factor_nav.csv')
    #print factor_nav_df.index[-1], factor_nav_df.iloc[-1]

    return
