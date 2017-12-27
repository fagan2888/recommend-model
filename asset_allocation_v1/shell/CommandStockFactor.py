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
import LabelAsset
import EqualRiskAssetRatio
import EqualRiskAsset
import HighLowRiskAsset
import os
import DBData
import AllocationData
import time
import RiskHighLowRiskAsset
import ModelHighLowRisk
import GeneralizationPosition
import Const
import WeekFund2DayNav
import FixRisk
import DFUtil
import LabelAsset
import Financial as fin

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund
from db.asset_stock_factor import *
from db.asset_stock import *

import traceback, code

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
def insert_factor_info(ctx, optfilepath):
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



def all_stock_info():

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    all_stocks = pd.read_sql(session.query(ra_stock.globalid, ra_stock.sk_secode).statement, session.bind, index_col = ['sk_secode'])
    session.commit()
    session.close()

    return all_stocks


def stock_st():

    all_stocks = all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_sk_specialtrade.secode, tq_sk_specialtrade.selecteddate,
            tq_sk_specialtrade.outdate).filter(tq_sk_specialtrade.selectedtype <= 3).filter(tq_sk_specialtrade.secode.in_(set(all_stocks.index))).statement
    st_stocks = pd.read_sql(sql, session.bind, index_col = ['secode'], parse_dates = ['selecteddate', 'outdate'])
    session.commit()
    session.close()

    st_stocks = pd.merge(st_stocks, all_stocks, left_index=True, right_index=True)

    return st_stocks


@sf.command()
@click.pass_context
def stock_factor(ctx):

    '''compute stock factor value
    '''
    #ln_price_df = ln_price_factor()
    #highlow_price_factor()
    relative_strength_factor()


def ln_price_factor():

    all_stocks = all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    data = {}
    for i in range(0, len(all_stocks.index) - 3400):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print globalid, secode
        sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.tclose).filter(tq_qt_skdailyprice.secode == secode).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate']).replace(0.0, np.nan)
        quotation = quotation.sort_index()
        data[globalid] = np.log(quotation.tclose)

    session.commit()
    session.close()

    ln_price_df = pd.DataFrame(data)

    return ln_price_df


def highlow_price_factor():

    all_stocks = all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_12m = {}
    data_6m = {}
    data_3m = {}
    data_1m = {}

    for i in range(0, len(all_stocks.index) - 3300):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print globalid, secode
        sql = session.query(tq_sk_dquoteindic.tradedate, tq_sk_dquoteindic.tcloseaf).filter(tq_sk_dquoteindic.secode == secode).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
        quotation = quotation.sort_index()
        quotation = quotation.reindex(pd.date_range(quotation.index.min(), quotation.index.max()))
        tclose_12m_max = quotation.tcloseaf.rolling(360, 1).max().iloc[360:]
        tclose_6m_max = quotation.tcloseaf.rolling(180, 1).max().iloc[180:]
        tclose_3m_max = quotation.tcloseaf.rolling(90, 1).max().iloc[90:]
        tclose_1m_max = quotation.tcloseaf.rolling(30, 1).max().iloc[30:]
        tclose_12m_min = quotation.tcloseaf.rolling(360, 1).min().iloc[360:]
        tclose_6m_min = quotation.tcloseaf.rolling(180, 1).min().iloc[180:]
        tclose_3m_min = quotation.tcloseaf.rolling(90, 1).min().iloc[90:]
        tclose_1m_min = quotation.tcloseaf.rolling(30, 1).min().iloc[30:]
        data_12m[globalid] = tclose_12m_max / tclose_12m_min
        data_6m[globalid] = tclose_6m_max / tclose_6m_min
        data_3m[globalid] = tclose_3m_max / tclose_3m_min
        data_1m[globalid] = tclose_1m_max / tclose_1m_min

    session.commit()
    session.close()

    highlow_price_12m_df = pd.DataFrame(data_12m)
    highlow_price_6m_df = pd.DataFrame(data_6m)
    highlow_price_3m_df = pd.DataFrame(data_3m)
    highlow_price_1m_df = pd.DataFrame(data_1m)

    return highlow_price_12m_df, highlow_price_6m_df, highlow_price_3m_df, highlow_price_1m_df


def relative_strength_factor():

    all_stocks = all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    data_6m = {}
    data_3m = {}
    data_1m = {}
    data    = {}

    for i in range(0, len(all_stocks.index) - 3300):
        secode = all_stocks.index[i]
        globalid = all_stocks.loc[secode, 'globalid']
        print globalid, secode
        sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.Yield, tq_sk_yieldindic.yieldm, tq_sk_yieldindic.yield3m, tq_sk_yieldindic.yield6m).filter(tq_sk_yieldindic.secode == secode).statement
        quotation = pd.read_sql(sql, session.bind , index_col = ['tradedate'], parse_dates = ['tradedate'])
        quotation = quotation.sort_index()
        quotation = quotation / 100.0
        print quotation.head()

        data_6m[globalid] = quotation.yield6m
        data_3m[globalid] = quotation.yield3m
        data_1m[globalid] = quotation.yield1m
        data[globalid] = quotation.Yield


    relative_strength_6m_df = pd.DataFrame(data_6m)
    relative_strength_3m_df = pd.DataFrame(data_3m)
    relative_strength_1m_df = pd.DataFrame(data_1m)
    relative_strength_df = pd.DataFrame(data_1m)
