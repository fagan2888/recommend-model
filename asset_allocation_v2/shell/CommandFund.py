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
import os
import DBData
import time
import Const
import DFUtil
import LabelAsset

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from tabulate import tabulate
from db import database, base_ra_fund_nav, base_ra_index, base_ra_fund, base_ra_index_nav
import MySQLdb

import traceback, code

logger = logging.getLogger(__name__)

@click.group()
@click.pass_context
def fund(ctx):
    '''fund pool group
    '''
    pass

@fund.command()
@click.pass_context
def nofee_fund(ctx):
    conn  = MySQLdb.connect(**config.db_base)
    conn.autocommit(True)
    sql = 'select fi_code, fi_name, fi_yingmi_subscribe_status from fund_infos'
    all_fund = pd.read_sql(sql, conn, index_col = ['fi_code']).dropna()
    sql = 'select ff_code, ff_type, ff_fee from fund_fee'
    fee_fund = pd.read_sql(sql, conn, index_col = ['ff_code'])
    no_fee_codes = []
    for k, v in fee_fund.groupby(level = [0]):
        if 5 not in v.ff_type.ravel():
            no_fee_codes.append(k)
    codes = fee_fund.loc[no_fee_codes].index & all_fund.index
    no_fee_fund = all_fund.loc[codes].drop_duplicates()
    sql = 'select ra_code from ra_fund where ra_type = 1'
    stock_fund = pd.read_sql(sql, conn, index_col = ['ra_code'])
    no_fee_fund = no_fee_fund.loc[stock_fund.index & no_fee_fund.index]
    print(no_fee_fund)
    conn.close()


@fund.command()
@click.option('--id', 'optid', help='specify ra corr id (e.g. 500001,500002')
@click.option('--fund', 'optfund', help='specify fund code (e.g. 519983,213009')
@click.option('--list/--no-list', 'optlist', default=False, help='list corr to update')
@click.pass_context
def corr(ctx, optid, optfund, optlist):
    ''' calc pool corr
    '''
    corrs = None
    if optid is not None:
        corrs = optid.split(',')

    codes = None
    if optfund is not None:
        codes = optfund.split(',')

    df_corr = load_ra_corr(corrs)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_corr['ra_name'] = df_corr['ra_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_corr, headers='keys', tablefmt='psql'))
        return 0

    for _, corr in df_corr.iterrows():
        corr_update(corr, codes)

def corr_update(corr, codes):
    ra_index = base_ra_index.find(corr['ra_index_id'])
    if ra_index is None:
        click.echo(click.style(
            "unknown index [%s]for calc corr!" % (corr['ra_index_id']), fg="yellow"))
        return False

    yesterday = (datetime.now() - timedelta(days=1));
    enddate = yesterday.strftime("%Y-%m-%d")

    #
    # 加载指数数据
    #
    index_code = ra_index['ra_code']
    if corr['ra_date_type'] == 1:
        df_nav_index = DBData.db_index_value_daily('2015-10-08', enddate, codes=[index_code])
    else:
        df_nav_index = DBData.db_index_value('2015-10-08', enddate, codes=[index_code])
    df_inc_index = df_nav_index.pct_change().fillna(0.0)

    #
    # 加载基金列表
    #
    df_fund = base_ra_fund.load(codes=codes)

    data = []
    with click.progressbar(length=len(df_fund.index), label=('update corr for corr %d' % (corr['globalid'])).ljust(30)) as bar:
        for _,fund in df_fund.iterrows():
            bar.update(1)
            tmp = corr_update_fund(corr, fund, df_inc_index)
            if tmp is not None:
                data.append([
                    corr['globalid'],
                    fund['globalid'],
                    fund['ra_code'],
                    "%.4f" % (tmp),
                ])

    df_new = pd.DataFrame(data, columns=['ra_corr_id', 'ra_fund_id', 'ra_fund_code', 'ra_corr'])
    df_new = df_new.set_index(['ra_corr_id', 'ra_fund_id'])

    db = database.connection('base')
    # 加载旧数据
    t2 = Table('ra_corr_fund', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.ra_corr_id,
        t2.c.ra_fund_id,
        t2.c.ra_fund_code,
        t2.c.ra_corr,
    ]
    stmt_select = select(columns2, (t2.c.ra_corr_id == corr['globalid']))
    if codes is not None:
        stmt_select = stmt_select.where(t2.c.ra_fund_code.in_(codes))

    df_old = pd.read_sql(stmt_select, db, index_col=['ra_corr_id', 'ra_fund_id'])
    if not df_old.empty:
        df_old['ra_corr'] = df_old['ra_corr'].map("{:.4f}".format)

    # 更新数据库
    database.batch(db, t2, df_new, df_old, timestamp=True)

def corr_update_fund(corr, fund, df_inc_index):
    yesterday = (datetime.now() - timedelta(days=1));
    enddate = yesterday.strftime("%Y-%m-%d")
    #
    # 加载基金数据
    #
    if corr['ra_date_type'] == 1:
        # df_nav_fund = DBData.db_fund_value_daily('2015-10-08', enddate, codes=[fund['ra_code']])
        df_nav_fund = base_ra_fund_nav.load_daily('2015-10-08', enddate, codes=[fund['ra_code']])
    else:
        # df_nav_fund = DBData.db_fund_value('2015-10-08', enddate, codes=[fund['ra_code']])
        df_nav_fund = base_ra_fund_nav.load_weekly('2015-10-08', enddate, codes=[fund['ra_code']])
    if df_nav_fund.empty:
        logger.warn("missing nav for fund [id: %d, code:%s]", fund['globalid'], fund['ra_code'])
        return None

    df_inc_fund = df_nav_fund.pct_change().fillna(0.0)

    # print df_inc_index.head()
    # print fund, df_inc_fund.head()
    df_inc = pd.DataFrame({'ra_index':df_inc_index.iloc[:, 0], 'ra_fund':df_inc_fund.ix[df_inc_index.index, 0]})
    df_inc.fillna(0.0, inplace=True)
    df_corr = df_inc.corr()
    df_corr.fillna(0.0, inplace=True)

    if df_corr.empty:
        corr = 0.0
    else:
        corr = df_corr.loc['ra_index', 'ra_fund']

    return corr


def load_ra_corr(corrs):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('ra_corr', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.ra_index_id,
        t.c.ra_date_type,
        t.c.ra_lookback,
        t.c.ra_name,
    ]

    s = select(columns)
    if corrs is not None:
        s = s.where(t.c.globalid.in_(corrs))

    df = pd.read_sql(s, db)

    return df

@fund.command(name='type')
@click.option('--id', 'optid', help='specify type id (e.g. 1001,1002')
@click.option('--fund', 'optfund', help='specify fund code (e.g. 519983,213009')
@click.option('--list/--no-list', 'optlist', default=False, help='list fund to update')
@click.pass_context
def type_command(ctx, optid, optfund, optlist):
    ''' calc fund type base on corr
    '''
    types = None
    if optid is not None:
        types = optid.split(',')

    codes = None
    if optfund is not None:
        codes = optfund.split(',')

    df_corr = load_ra_corr(corrs)

    if optlist:
        #print df_pool
        #df_pool.reindex_axis(['ra_type','ra_date_type', 'ra_fund_type', 'ra_lookback', 'ra_name'], axis=1)
        df_corr['ra_name'] = df_corr['ra_name'].map(lambda e: e.decode('utf-8'))
        print(tabulate(df_corr, headers='keys', tablefmt='psql'))
        return 0

    for _, corr in df_corr.iterrows():
        corr_update(corr, codes)



@fund.command()
@click.pass_context
def similar_corr_fund(ctx):
    start_date = '2018-06-01'
    end_date = '2019-06-01'
    index = DBData.trade_date_lookback_index(end_date=end_date, lookback=52)
    df_nav_bond = DBData.bond_fund_value(start_date, end_date)
    df_nav_bond = df_nav_bond.reindex(index, method='pad')
    df_nav_bond = df_nav_bond.dropna()
    df_inc_bond = df_nav_bond.pct_change().fillna(0.0)
    corr = df_inc_bond.corr()
    corr = corr.loc['519782']
    corr = corr.sort_values(ascending=False)
    print(corr.head(100))
