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
from sqlalchemy import MetaData, Table, select, func, literal_column
from tabulate import tabulate
from db import database, base_exchange_rate_index, base_ra_index, asset_ra_pool_fund, base_ra_fund, asset_ra_pool
from util import xdict

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def analysis(ctx):
    '''
        analysis something
    '''
    pass


@analysis.command()
@click.option('--portfolio-id', 'optid', default=True, help=u'portfolio_id')
@click.option('--start-date', 'optstartdate', default=None, help=u'portfolio pos startdate')
@click.option('--end-date', 'optenddate', default=None, help=u'portfolio pos endate')
@click.pass_context
def portfolio_turnover_freq(ctx, optid, optstartdate, optenddate):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    portfolio_alloc_t = Table('ra_portfolio_alloc', metadata, autoload=True)
    portfolio_pos_t = Table('ra_portfolio_pos', metadata, autoload=True)

    alloc_columns = [
                    portfolio_alloc_t.c.globalid,
            ]

    s = select(alloc_columns).where(portfolio_alloc_t.c.ra_portfolio_id == optid.strip())

    alloc_df = pd.read_sql(s, db)

    pos_columns = [
                portfolio_pos_t.c.ra_date,
                portfolio_pos_t.c.ra_fund_code,
                portfolio_pos_t.c.ra_fund_ratio,
            ]


    nums = []
    for globalid in alloc_df['globalid'].ravel():
        s = select(pos_columns).where(portfolio_pos_t.c.ra_portfolio_id == globalid)
        pos_df = pd.read_sql(s, db, index_col = ['ra_date', 'ra_fund_code'], parse_dates = ['ra_date'])
        pos_df = pos_df.groupby(level=[0,1]).sum()
        pos_df = pos_df.unstack(1).fillna(0.0)
        dates = pos_df.index.ravel()
        if optstartdate is not None:
            pos_df = pos_df[pos_df.index >= optstartdate]
        if optenddate is not None:
            pos_df = pos_df[pos_df.index <= optenddate]
        #print globalid, ' change position num ' ,len(pos_df.index)
        nums.append(len(pos_df.index))


    change_position_num_df = pd.DataFrame(nums, index = alloc_df['globalid'], columns = ['num'])
    print change_position_num_df
    change_position_num_df.to_csv('tmp/' + str(optid.strip()) + '_change_position_num.csv')


@analysis.command()
@click.option('--new-portfolio-id', 'optnewid', default='', help=u'portfolio_new_id')
@click.option('--old-portfolio-id', 'optoldid', default='', help=u'portfolio_old_id')
@click.option('--fee', 'optfee', default=8, help=u'portfolio_fee')
@click.option('--start-date', 'optstartdate', default=None, help=u'portfolio pos startdate')
@click.option('--end-date', 'optenddate', default=None, help=u'portfolio pos endate')
@click.pass_context
def portfolio_compara(ctx, optnewid, optoldid, optfee, optstartdate, optenddate):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    portfolio_nav_t = Table('ra_portfolio_nav', metadata, autoload=True)

    nav_columns = [
                    portfolio_nav_t.c.ra_date,
                    portfolio_nav_t.c.ra_nav,
            ]

    s = select(nav_columns).where(portfolio_nav_t.c.ra_portfolio_id == optnewid.strip()).where(portfolio_nav_t.c.ra_type == optfee)
    new_nav_df = pd.read_sql(s, db, index_col = ['ra_date'], parse_dates = ['ra_date'])

    s = select(nav_columns).where(portfolio_nav_t.c.ra_portfolio_id == optoldid.strip()).where(portfolio_nav_t.c.ra_type == optfee)
    old_nav_df = pd.read_sql(s, db, index_col = ['ra_date'], parse_dates = ['ra_date'])


    old_nav_df = old_nav_df.reindex(new_nav_df.index).fillna(method = 'pad').fillna(1.0)


    new_nav = new_nav_df['ra_nav'].ravel()
    old_nav = old_nav_df['ra_nav'].ravel()


    interval = 7
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print '平均每周多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old)
    print '周战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old)


    interval = 30
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print '平均每月多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old)
    print '月战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old)


    interval = 60
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print '平均每两月多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old)
    print '两月战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old)


    interval = 90
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print '平均每季度多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old)
    print '季度战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old)


    interval = 180
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print '平均每半年多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old)
    print '半年战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old)


    interval = 360
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print '平均每年多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old)
    print '年战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old)


@analysis.command()
@click.option('--pool-id', 'optpool', default=True, help=u'pool id')
@click.pass_context
def fund_pool_info(ctx, optpool):


    db = database.connection('base')
    metadata = MetaData(bind=db)
    fund_info_t = Table('fund_infos', metadata, autoload=True)
    company_t = Table('company_infos', metadata, autoload=True)

    fund_info = [
            fund_info_t.c.fi_globalid,
            fund_info_t.c.fi_code,
            fund_info_t.c.fi_name,
            fund_info_t.c.fi_company_id,
    ]

    fund_info_df = pd.read_sql(select(fund_info), db)

    company_info = [
            company_t.c.ci_globalid,
            company_t.c.ci_name,
    ]


    company_info_df = pd.read_sql(select(company_info), db)


    fund_company_df = pd.merge(fund_info_df, company_info_df, left_on = ['fi_company_id'], right_on = ['ci_globalid'])
    fund_company_df = fund_company_df[['fi_globalid','fi_code', 'fi_name', 'ci_name']]
    fund_company_df = fund_company_df.set_index(['fi_globalid'])




    pool_id = optpool.strip()
    pool_name = asset_ra_pool.find(pool_id)['ra_name']
    pool_df = asset_ra_pool_fund.load(pool_id)
    pool_df = pool_df['ra_fund_code']
    pool_df = pool_df.reset_index()
    pool_df = pool_df.set_index('ra_fund_id')
    #print fund_company_df.head()
    #print pool_df.head()


    pool_fund_company_df = pd.concat([pool_df, fund_company_df], axis = 1, join_axes = [pool_df.index])
    pool_fund_company_df = pool_fund_company_df[['ra_date','ra_fund_code','fi_name','ci_name']]
    pool_fund_company_df['pool_name'] = pool_name
    #pool_fund_company_df = pool_fund_company_df[pool_fund_company_df['ra_date'] >= '2016-10-02']
    print pool_fund_company_df
    pool_fund_company_df.to_csv('pool_fund_company_df.csv', encoding = 'gbk')



    '''
    fund_info_df = base_ra_fund.load()
    fund_info_df = fund_info_df[['globalid', 'ra_code', 'ra_name']]
    fund_info_df = fund_info_df.set_index('globalid')
    print fund_info_df
    '''
