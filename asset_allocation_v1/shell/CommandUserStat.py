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


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func, literal_column
from tabulate import tabulate
from db import database, base_exchange_rate_index, base_ra_index, asset_ra_pool_fund, base_ra_fund, asset_ra_pool, asset_on_online_nav
from util import xdict

import traceback, code


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def user_stat(ctx):
    '''
        analysis something
    '''
    pass


@user_stat.command()
@click.pass_context
def user_label(ctx):

    db = database.connection('asset')
    metadata = MetaData(bind=db)

    user_account_infos_t = Table('user_account_infos', metadata, autoload=True)

    user_account_columns = [
                    user_account_infos_t.c.ua_uid,
                    user_account_infos_t.c.ua_phone,
                    user_account_infos_t.c.ua_name,
                    user_account_infos_t.c.ua_service_id,
            ]

    s = select(user_account_columns)

    user_account_df = pd.read_sql(s, db, index_col = ['ua_uid'])


    db = database.connection('trade')
    metadata = MetaData(bind=db)

    ts_share_fund_t = Table('ts_share_fund', metadata, autoload=True)
    ts_share_fund_columns = [
                    ts_share_fund_t.c.ts_uid,
                    ts_share_fund_t.c.ts_amount,
            ]

    s = select(ts_share_fund_columns)
    ts_share_fund_df = pd.read_sql(s, db, index_col = ['ts_uid'])
    user_holding_asset = ts_share_fund_df.groupby(ts_share_fund_df.index).sum()
    user_holding_asset = user_holding_asset[user_holding_asset['ts_amount'] >= 10.0]


    db = database.connection('base')
    trade_dates = pd.read_sql('select td_date from trade_dates', db, parse_dates = ['td_date'], index_col = ['td_date'])
    trade_dates = trade_dates.sort_index(ascending = False)
    trade_dates = trade_dates[trade_dates.index <= datetime.now()]
    trade_dates = trade_dates.index.ravel()
    start_date = trade_dates[2]


    db = database.connection('trade')
    metadata = MetaData(bind=db)
    ts_order_t = Table('ts_order', metadata, autoload=True)

    ts_order_columns = [
                ts_order_t.c.ts_uid,
                ts_order_t.c.ts_trade_type,
                ts_order_t.c.ts_placed_date,
                ts_order_t.c.ts_placed_amount,
                ts_order_t.c.ts_placed_percent,
            ]


    s = select(ts_order_columns).where( (ts_order_t.c.ts_trade_type == 3) | (ts_order_t.c.ts_trade_type == 4) ).where((ts_order_t.c.ts_trade_status == 1) | (ts_order_t.c.ts_trade_status == 5) | (ts_order_t.c.ts_trade_status == 6)).where(ts_order_t.c.ts_placed_date >= start_date)
    ts_order_df = pd.read_sql(s, db, index_col = ['ts_uid'], parse_dates = ['ts_placed_date'])
    ts_order_buy_uids = set(ts_order_df[ts_order_df['ts_trade_type'] == 3].index)
    ts_order_redeem_uids = set(ts_order_df[ts_order_df['ts_trade_type'] == 4].index)
    #print ts_order_buy_uids
    #print ts_order_redeem_uids


    db = database.connection('portfolio_sta')
    metadata = MetaData(bind=db)
    ds_order_t = Table('ds_order_pdate', metadata, autoload=True)

    ds_order_columns = [
                ds_order_t.c.ds_uid,
                ds_order_t.c.ds_placed_date,
            ]

    s = select(ds_order_columns).where(ds_order_t.c.ds_trade_type == 10)
    ds_order_df = pd.read_sql(s, db, index_col = ['ds_uid'], parse_dates = ['ds_placed_date'])
    #print ds_order_df
    #print ts_order_buy_uids
    #print ts_order_redeem_uids



    db = database.connection('tongji')
    metadata = MetaData(bind=db)

    log_raw_apps_t = Table('log_raw_apps', metadata, autoload=True)

    log_raw_apps_columns = [

            log_raw_apps_t.c.lr_uid,
            log_raw_apps_t.c.lr_date,
            log_raw_apps_t.c.lr_page,
            log_raw_apps_t.c.lr_ctrl,
            log_raw_apps_t.c.lr_ev,
            log_raw_apps_t.c.lr_ref,

            ]

    uid_d = 1000000000
    uid_u = 1999999999

    s = select(log_raw_apps_columns).where(log_raw_apps_t.c.lr_date >= start_date).where(log_raw_apps_t.c.lr_uid >= uid_d).where(log_raw_apps_t.c.lr_uid <= uid_u)
    log_raw_apps_df = pd.read_sql(s, db, index_col = ['lr_uid'])


    click_buy = []
    click_redeem = []
    for uid, group in log_raw_apps_df.groupby(log_raw_apps_df.index):
        for i in range(0 ,len(group)):
            record = group.iloc[i]
            #buy
            if record['lr_ref'] == 2013 and record['lr_ctrl'] == 4:
                if uid not in ts_order_buy_uids:
                    click_buy.append(uid)
            elif record['lr_page'] == 2006 and record['lr_ctrl'] == 4:
                if uid not in ts_order_buy_uids:
                    click_buy.append(uid)
            elif record['lr_page'] == 2013 and record['lr_ctrl'] == 6:
                if uid not in ts_order_redeem_uids:
                    click_redeem.append(uid)

    #print len(click_buy)
    #print len(click_redeem)


    db = database.connection('trade')

    ts_order_columns = [
                ts_order_t.c.ts_uid,
                ts_order_t.c.ts_trade_type,
                ts_order_t.c.ts_placed_date,
                ts_order_t.c.ts_placed_amount,
                ts_order_t.c.ts_placed_percent,
            ]




    start_date = (datetime.now() - timedelta(35)).strftime('%Y-%m-%d')
    #print start_date
    s = select(ts_order_columns).where( (ts_order_t.c.ts_trade_type == 3) | (ts_order_t.c.ts_trade_type == 4) ).where((ts_order_t.c.ts_trade_status == 1) | (ts_order_t.c.ts_trade_status == 5) | (ts_order_t.c.ts_trade_status == 6)).where(ts_order_t.c.ts_placed_date >= start_date)
    ts_order_df = pd.read_sql(s, db, index_col = ['ts_uid'], parse_dates = ['ts_placed_date'])


    filter_uids = set()
    for uid , group in ts_order_df.groupby(ts_order_df.index):
        tmp_group = group[group['ts_trade_type'] == 3]
        last_record = group.iloc[-1]
        if last_record['ts_trade_type'] == 4 and last_record['ts_placed_percent'] == 1.0:
            filter_uids.add(uid)
        if len(tmp_group) > 0:
            filter_uids.add(uid)

    a_month_not_buy = set(user_holding_asset.index).difference(filter_uids)

    a_month_not_buy_df = pd.concat([user_account_df.loc[a_month_not_buy], user_holding_asset.loc[a_month_not_buy], ds_order_df.loc[a_month_not_buy]], axis = 1)
    #print a_month_not_buy_df
    click_buy_not_buy_df = pd.concat([user_account_df.loc[click_buy], user_holding_asset.loc[click_buy], ds_order_df.loc[click_buy]], axis = 1)
    click_redeem_not_redeem_df = pd.concat([user_account_df.loc[click_redeem], user_holding_asset.loc[click_redeem], ds_order_df.loc[click_redeem]], axis = 1)
    #print click_redeem_not_redeem_df
    a_month_not_buy_df.to_csv('a_month_not_buy.csv', encoding='gbk')
    click_buy_not_buy_df.to_csv('click_buy_not_buy.csv', encoding='gbk')
    click_redeem_not_redeem_df.to_csv('click_redeem_not_redeem.csv', encoding='gbk')
