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


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from tabulate import tabulate
from db import database, base_exchange_rate_index, base_ra_index
from util import xdict

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def util(ctx):
    '''
        import, update, cp portfolio highlow markowitz
    '''
    pass


@util.command()
@click.option('--path', 'optpath', default=True, help=u'file path')
@click.pass_context
def imp(ctx, optpath):


    '''
    all_portfolio_df = pd.read_csv(optpath.strip(), parse_dates = ['start_date', 'end_date'])

    portfolio_id                 = all_portfolio_df['ra_portfolio_id'].unique().item()
    markowitz_id                 = all_portfolio_df['mz_markowitz_id'].unique().item()
    highlow_id                   = all_portfolio_df['mz_highlow_id'].unique().item()
    portfolio_name               = all_portfolio_df['ra_portfolio_name'].unique().item()
    markowitz_name               = all_portfolio_df['mz_markowitz_name'].unique().item()
    highlow_name                 = all_portfolio_df['mz_highlow_name'].unique().item()
    allocate_algo                = all_portfolio_df['allocate_algo'].unique().item()
    turnover_filter              = all_portfolio_df['turnover_filter'].unique().item()
    date_type                    = all_portfolio_df['date_type'].unique().item()
    adjust_position_period       = all_portfolio_df['adjust_position_period'].unique().item()
    adjust_position_dates        = all_portfolio_df['adjust_position_dates'].unique().item()
    look_back                    = all_portfolio_df['look_back'].unique().item()
    start_dates                  = all_portfolio_df['start_date'].unique().ravel()
    end_dates                    = all_portfolio_df['end_date'].unique().ravel()
    start_dates.sort()
    start_date                   = start_dates[0]
    end_dates.sort()
    end_date                     = end_dates[0]
    portfolio_df = pd.DataFrame([[portfolio_id, allocate_algo, turnover_filter, date_type, adjust_position_period, adjust_position_dates, look_back,start_date, end_date]], columns = ['portfolio_id', 'allocate_algo', 'turnover_filter', 'date_type', 'adjust_position_period', 'adjust_position_dates', 'look_back','start_date', 'end_date'])
    portfolio_df = portfolio_df.set_index(['portfolio_id'])

    portfolio_asset_data = []
    for i in range(0, len(all_portfolio_df)):
        record = all_portfolio_df.iloc[i]
        portfolio_asset_data.append([record['risk'], record['asset_id'],record['pool_id'], record['sum1'], record['sum2'], record['lower'], record['upper'],
                                    record['wavelet'], record['start_date'], record['end_date'], record['timing_id'], record['riskmgr_id']])

    portfolio_asset_df = pd.DataFrame(portfolio_asset_data, columns = ['risk','asset_id','pool_id', 'sum1', 'sum2', 'lower', 'upper',
                                                                        'wavelet', 'start_date', 'end_date', 'timing_id', 'riskmgr_id'])

    print portfolio_df
    print portfolio_asset_df
    '''

    all_portfolio_df = pd.read_csv(optpath.strip(), parse_dates = ['start_date', 'end_date'])
    imp_markowitz(all_portfolio_df)
    imp_highlow(all_portfolio_df)


def imp_portfolio():

    df = df.copy()

    portfolio_id                 = df['mz_markowitz_id'].unique().item()
    portfolio_name               = df['mz_markowitz_name'].unique().item()
    markowitz_df = pd.DataFrame([[markowitz_id, markowitz_name]], columns = ['globalid', 'mz_name'])
    markowitz_df['mz_type'] = 1
    markowitz_df = markowitz_df.set_index(['globalid'])

    for k, v in df.groupby(['risk']):
        markowitz_id = v['mz_markowitz_id'].unique().item()
        risk = v['risk'].unique().item()
        markowitz_id_num = markowitz_id.strip().split('.')[1]

        markowitz_risk_id = markowitz_id.replace(markowitz_id_num, str(string.atoi(markowitz_id_num) + int(risk * 10) % 10))

        tmpv = v[['asset_id', 'allocate_algo','sum1', 'sum2', 'lower', 'upper', 'start_date', 'end_date']]
        tmpv['mz_markowitz_id'] = markowitz_risk_id

        tmpv = tmpv.rename(columns = {'asset_id':'mz_asset_id', 'allocate_algo':'mz_algo', 'sum1' : 'mz_sum1_limit',
                                        'sum2' : 'mz_sum2_limit','upper' : 'mz_upper_limit','lower' : 'mz_lower_limit',
                                        'start_date' : 'mz_start_date', 'end_date': 'mz_end_date'})

        tmpv['mz_markowitz_asset_id'] = tmpv['mz_asset_id']


        data = []
        for col in v.columns:
            key = col.strip()
            if key.startswith('allocate'):
                value = str(v[col].unique().item()).strip()
                value = value if not value == 'nan' else ''
                data.append([markowitz_risk_id, key, value])

        argv_df = pd.DataFrame(data, columns = ['mz_markowitz_id', 'mz_key', 'mz_value'])
        argv_df = argv_df.set_index(['mz_markowitz_id', 'mz_key'])

        #print argv_df


def imp_highlow(df):

    df = df.copy()

    highlow_id                 = df['mz_highlow_id'].unique().item()
    highlow_name               = df['mz_highlow_name'].unique().item()
    markowitz_id               = df['mz_markowitz_id'].unique().item()
    highlow_df = pd.DataFrame([[highlow_id, highlow_name, markowitz_id]], columns = ['globalid', 'mz_name', 'mz_markowitz_id'])
    highlow_df['mz_type'] = 1
    highlow_df = highlow_df.set_index(['globalid'])
    highlow_df['mz_algo'] = df['mz_highlow_algo'].unique().item()
    highlow_df['mz_persistent'] = 0


    for k, v in df.groupby(['risk']):
        highlow_id = v['mz_highlow_id'].unique().item()
        risk = v['risk'].unique().item()
        highlow_id_num = highlow_id.strip().split('.')[1]

        highlow_risk_id = highlow_id.replace(highlow_id_num, str(string.atoi(highlow_id_num) + int(risk * 10) % 10))

        tmpv = v[['asset_id', 'riskmgr_id', 'pool_id']]
        tmpv['mz_highlow_id'] = highlow_risk_id

        tmpv = tmpv.rename(columns = {'asset_id':'mz_asset_id', 'riskmgr_id':'mz_riskmgr_id', 'pool_id' : 'mz_pool_id'})

        tmpv['mz_asset_type'] = 0
        tmpv['mz_highlow_id'] = highlow_risk_id

        asset_names = []
        for asset_id in tmpv['mz_asset_id']:
            asset_names.append(find_asset_name(asset_id))
        tmpv['mz_asset_name'] = asset_names

        print tmpv

        data = []
        for col in v.columns:
            key = col.strip()
            if key.startswith('highlow'):
                value = str(v[col].unique().item()).strip()
                value = value if not value == 'nan' else ''
                data.append([highlow_risk_id, key, value])

        argv_df = pd.DataFrame(data, columns = ['mz_highlow_id', 'mz_key', 'mz_value'])
        argv_df = argv_df.set_index(['mz_highlow_id', 'mz_key'])

        #print argv_df

    pass


def imp_markowitz(df):

    df = df.copy()

    markowitz_id                 = df['mz_markowitz_id'].unique().item()
    markowitz_name               = df['mz_markowitz_name'].unique().item()
    markowitz_df = pd.DataFrame([[markowitz_id, markowitz_name]], columns = ['globalid', 'mz_name'])
    markowitz_df['mz_type'] = 1
    markowitz_df = markowitz_df.set_index(['globalid'])

    for k, v in df.groupby(['risk']):
        markowitz_id = v['mz_markowitz_id'].unique().item()
        risk = v['risk'].unique().item()
        markowitz_id_num = markowitz_id.strip().split('.')[1]

        markowitz_risk_id = markowitz_id.replace(markowitz_id_num, str(string.atoi(markowitz_id_num) + int(risk * 10) % 10))

        tmpv = v[['asset_id', 'allocate_algo','sum1', 'sum2', 'lower', 'upper', 'start_date', 'end_date']]
        tmpv['mz_markowitz_id'] = markowitz_risk_id

        tmpv = tmpv.rename(columns = {'asset_id':'mz_asset_id', 'allocate_algo':'mz_algo', 'sum1' : 'mz_sum1_limit',
                                        'sum2' : 'mz_sum2_limit','upper' : 'mz_upper_limit','lower' : 'mz_lower_limit',
                                        'start_date' : 'mz_start_date', 'end_date': 'mz_end_date'})

        tmpv['mz_markowitz_asset_id'] = tmpv['mz_asset_id']
        asset_names = []
        for asset_id in tmpv['mz_asset_id']:
            asset_names.append(find_asset_name(asset_id))
        tmpv['mz_asset_name'] = tmpv['mz_markowitz_asset_name'] = asset_names

        data = []
        for col in v.columns:
            key = col.strip()
            if key.startswith('allocate'):
                value = str(v[col].unique().item()).strip()
                value = value if not value == 'nan' else ''
                data.append([markowitz_risk_id, key, value])

        argv_df = pd.DataFrame(data, columns = ['mz_markowitz_id', 'mz_key', 'mz_value'])
        argv_df = argv_df.set_index(['mz_markowitz_id', 'mz_key'])

        #print argv_df


def find_asset_name(asset_id):
    if asset_id.strip().isdigit():
        #print int(asset_id)
        asset_id = int(asset_id)
        xtype = asset_id / 10000000
        if 12 == xtype:
            record = base_ra_index.find(asset_id)
            return record[2].strip()
        else:
            return None
    elif asset_id.strip().startswith('ERI'):
        record = base_exchange_rate_index.find(asset_id)
        return record[2].strip()
    else:
        return None
    #flag = asset_id.strip().split('.')[0]


@util.command()
@click.option('--from', 'optfrom', default=True, help=u'--from id')
@click.option('--to', 'optto', default=True, help=u'--to id')
@click.option('--name', 'optname', default=True, help=u'name')
@click.pass_context
def cp(ctx, optfrom, optto, optname):
    pass
