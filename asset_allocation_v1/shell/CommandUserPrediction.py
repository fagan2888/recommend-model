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
from sqlalchemy import *
from sqlalchemy import MetaData, Table, select, func, literal_column
from tabulate import tabulate
from db import database, base_exchange_rate_index, base_ra_index
from util import xdict
from sklearn import preprocessing

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def user(ctx):
    '''
        analysis something
    '''
    pass


@user.command()
@click.pass_context
def export_app_log(ctx):

    db_tongji_uri =  'mysql://root:Mofang123@127.0.0.1/tmp_tongji?charset=utf8&use_unicode=0'
    db = create_engine(db_tongji_uri)
    metadata = MetaData(bind=db)

    t = Table('log_raw_apps', metadata, autoload=True)
    columns = [
        t.c.lr_uid,
        t.c.lr_date,
        t.c.lr_time,
        t.c.lr_page,
        t.c.lr_ctrl,
        t.c.lr_ref,
        t.c.lr_ev,
        t.c.lr_ts,
    ]

    uid_d = 1000000154
    uid_u = 1999999999

    s = select(columns).where(t.c.lr_uid >= uid_d).where(t.c.lr_uid <= uid_u)
    df = pd.read_sql(s, db)

    print df.head()
    df.to_csv('app_log.csv')

    return df


@user.command()
@click.pass_context
def export_ts_holding_nav(ctx):

    db_ts_holding_nav_uri =  'mysql://root:Mofang123@127.0.0.1/trade?charset=utf8&use_unicode=0'
    db = create_engine(db_ts_holding_nav_uri)
    metadata = MetaData(bind=db)

    t = Table('ts_holding_nav', metadata, autoload=True)
    columns = [
        t.c.ts_uid,
        t.c.ts_portfolio_id,
        t.c.ts_date,
        t.c.ts_nav,
        t.c.ts_share,
        t.c.ts_asset,
        t.c.ts_processing_asset,
        t.c.ts_profit,
    ]

    uid_d = 1000000154
    uid_u = 1999999999

    s = select(columns).where(t.c.ts_uid >= uid_d).where(t.c.ts_uid <= uid_u)
    df = pd.read_sql(s, db)

    print df.head()
    df.to_csv('ts_holding_nav.csv')

    return df

@user.command()
@click.pass_context
def export_ts_order(ctx):

    db_ts_order_uri =  'mysql://root:Mofang123@127.0.0.1/trade?charset=utf8&use_unicode=0'
    db = create_engine(db_ts_order_uri)
    metadata = MetaData(bind=db)

    t = Table('ts_order', metadata, autoload=True)
    columns = [
        t.c.ts_txn_id,
        t.c.ts_uid,
        t.c.ts_portfolio_id,
        t.c.ts_trade_type,
        t.c.ts_trade_status,
        t.c.ts_trade_date,
        t.c.ts_placed_date,
        t.c.ts_placed_time,
        t.c.ts_placed_amount,
        t.c.ts_placed_percent,
        t.c.ts_acked_amount,
        t.c.ts_risk,
        t.c.ts_invest_plan_id,
        t.c.ts_origin,
    ]

    uid_d = 1000000154
    uid_u = 1999999999

    s = select(columns).where(t.c.ts_uid >= uid_d).where(t.c.ts_uid <= uid_u)
    df = pd.read_sql(s, db)

    print df.head()
    df.to_csv('ts_order.csv')

    return df


@user.command()
#@click.option('--logfile', 'optlogfile', default=True, help=u'logfile path')
@click.pass_context
def analysis_feature(ctx):

    #log_df = pd.read_csv('user_prediction/app_log.csv')
    holding_df = pd.read_csv('user_prediction/ts_holding_nav.csv')
    order_df = pd.read_csv('user_prediction/ts_order.csv')

    #print order_df.columns
    order_cols = ['ts_uid', 'ts_trade_type', 'ts_trade_status', 'ts_trade_date', 'ts_placed_date', 'ts_placed_time',
                            'ts_placed_amount','ts_placed_percent', 'ts_acked_amount', 'ts_risk']
    order_df = order_df[order_cols]
    order_df = order_df.dropna()
    order_df = order_df.set_index(['ts_uid'])
    #order_df = order_df[order_df['ts_trade_status'] > 0]
    order_df = order_df[order_df['ts_trade_type'] < 7]
    order_df = order_df[order_df['ts_acked_amount'] >= 100]
    #order_df = order_df[order_df['ts_trade_type'] == 4]
    #order_df = order_df['ts_invest_plan_id']
    #print order_df


    holding_cols = ['ts_uid', 'ts_date', 'ts_nav', 'ts_share', 'ts_asset', 'ts_processing_asset', 'ts_profit']
    holding_df = holding_df[holding_cols]
    holding_df = holding_df.set_index(['ts_uid'])
    #print holding_df.head()

    #holding_order_df = pd.concat([holding_df, order_df], axis = 1, join_axes = [holding_df.index])
    #print holding_order_df

    feats = []

    for uid, holding_group in holding_df.groupby(holding_df.index):

        print uid

        #print holding_group.head()
        holding_group = holding_group.reset_index()
        holding_group = holding_group.set_index(['ts_date'])

        if uid not in set(order_df.index):
            continue
        order_group = order_df.loc[uid]
        if not isinstance(order_group, pd.DataFrame):
            order_group = order_group.to_frame().T
        order_group = order_group.reset_index()
        order_group = order_group.set_index(['ts_placed_date'])
        #print order_group
        order_dates = list(order_group.index)
        order_dates.sort()
        order_group = order_group.loc[order_dates]

        start_date = order_dates[0]

        holding_group = holding_group[holding_group.index >= start_date]
        end_date = order_dates[-1]
        if 4 in order_group.loc[end_date, 'ts_trade_type'].ravel():
            holding_group = holding_group[holding_group.index <= end_date]

        #print holding_group.tail(10)

        dates = holding_group.index
        holding_group['ts_r'] = holding_group['ts_nav'].pct_change().fillna(0.0)

        holding_group['ts_180r'] = holding_group['ts_r'].rolling(180).sum()
        holding_group['ts_90r'] = holding_group['ts_r'].rolling(90).sum()
        holding_group['ts_60r'] = holding_group['ts_r'].rolling(60).sum()
        holding_group['ts_30r'] = holding_group['ts_r'].rolling(30).sum()
        holding_group['ts_15r'] = holding_group['ts_r'].rolling(15).sum()
        holding_group['ts_7r'] = holding_group['ts_r'].rolling(7).sum()

        holding_group['ts_180std'] = holding_group['ts_r'].rolling(180).std()
        holding_group['ts_90std'] = holding_group['ts_r'].rolling(90).std()
        holding_group['ts_60std'] = holding_group['ts_r'].rolling(60).std()
        holding_group['ts_30std'] = holding_group['ts_r'].rolling(30).std()
        holding_group['ts_15std'] = holding_group['ts_r'].rolling(15).std()
        holding_group['ts_7std'] = holding_group['ts_r'].rolling(7).std()

        holding_group['ts_180share_chg'] = holding_group['ts_share'].rolling(180).apply(lambda x : x[-1] / x[0])
        holding_group['ts_90share_chg'] = holding_group['ts_share'].rolling(90).apply(lambda x : x[-1] / x[0])
        holding_group['ts_60share_chg'] = holding_group['ts_share'].rolling(60).apply(lambda x : x[-1] / x[0])
        holding_group['ts_30share_chg'] = holding_group['ts_share'].rolling(30).apply(lambda x : x[-1] / x[0])
        holding_group['ts_15share_chg'] = holding_group['ts_share'].rolling(15).apply(lambda x : x[-1] / x[0])
        holding_group['ts_7share_chg'] = holding_group['ts_share'].rolling(7).apply(lambda x : x[-1] / x[0])

        holding_group['ts_180share_diff'] = holding_group['ts_share'].rolling(180).apply(lambda x : x[-1] - x[0])
        holding_group['ts_90share_diff'] = holding_group['ts_share'].rolling(90).apply(lambda x : x[-1] - x[0])
        holding_group['ts_60share_diff'] = holding_group['ts_share'].rolling(60).apply(lambda x : x[-1] - x[0])
        holding_group['ts_30share_diff'] = holding_group['ts_share'].rolling(30).apply(lambda x : x[-1] - x[0])
        holding_group['ts_15share_diff'] = holding_group['ts_share'].rolling(15).apply(lambda x : x[-1] - x[0])
        holding_group['ts_7share_diff'] = holding_group['ts_share'].rolling(7).apply(lambda x : x[-1] - x[0])

        holding_group['ts_180asset_diff'] = holding_group['ts_asset'].rolling(180).apply(lambda x : x[-1] - x[0])
        holding_group['ts_90asset_diff'] = holding_group['ts_asset'].rolling(90).apply(lambda x : x[-1] - x[0])
        holding_group['ts_60asset_diff'] = holding_group['ts_asset'].rolling(60).apply(lambda x : x[-1] - x[0])
        holding_group['ts_30asset_diff'] = holding_group['ts_asset'].rolling(30).apply(lambda x : x[-1] - x[0])
        holding_group['ts_15asset_diff'] = holding_group['ts_asset'].rolling(15).apply(lambda x : x[-1] - x[0])
        holding_group['ts_7asset_diff'] = holding_group['ts_asset'].rolling(7).apply(lambda x : x[-1] - x[0])

        holding_group['ts_180asset_chg'] = holding_group['ts_asset'].rolling(180).apply(lambda x : x[-1] / x[0])
        holding_group['ts_90asset_chg'] = holding_group['ts_asset'].rolling(90).apply(lambda x : x[-1] / x[0])
        holding_group['ts_60asset_chg'] = holding_group['ts_asset'].rolling(60).apply(lambda x : x[-1] / x[0])
        holding_group['ts_30asset_chg'] = holding_group['ts_asset'].rolling(30).apply(lambda x : x[-1] / x[0])
        holding_group['ts_15asset_chg'] = holding_group['ts_asset'].rolling(15).apply(lambda x : x[-1] / x[0])
        holding_group['ts_7asset_chg'] = holding_group['ts_asset'].rolling(7).apply(lambda x : x[-1] / x[0])

        #print holding_group.tail(10)
        holding_group['ts_180profit_chg'] = holding_group['ts_profit'].rolling(180).apply(lambda x : x[-1] / x[0])
        holding_group['ts_90profit_chg'] = holding_group['ts_profit'].rolling(90).apply(lambda x : x[-1] / x[0])
        holding_group['ts_60profit_chg'] = holding_group['ts_profit'].rolling(60).apply(lambda x : x[-1] / x[0])
        holding_group['ts_30profit_chg'] = holding_group['ts_profit'].rolling(30).apply(lambda x : x[-1] / x[0])
        holding_group['ts_15profit_chg'] = holding_group['ts_profit'].rolling(15).apply(lambda x : x[-1] / x[0])
        holding_group['ts_7profit_chg'] = holding_group['ts_profit'].rolling(7).apply(lambda x : x[-1] / x[0])

        holding_group['ts_180profit_diff'] = holding_group['ts_profit'].rolling(180).apply(lambda x : x[-1] - x[0])
        holding_group['ts_90profit_diff'] = holding_group['ts_profit'].rolling(90).apply(lambda x : x[-1] - x[0])
        holding_group['ts_60profit_diff'] = holding_group['ts_profit'].rolling(60).apply(lambda x : x[-1] - x[0])
        holding_group['ts_30profit_diff'] = holding_group['ts_profit'].rolling(30).apply(lambda x : x[-1] - x[0])
        holding_group['ts_15profit_diff'] = holding_group['ts_profit'].rolling(15).apply(lambda x : x[-1] - x[0])
        holding_group['ts_7profit_diff'] = holding_group['ts_profit'].rolling(7).apply(lambda x : x[-1] - x[0])

        holding_group['ts_profit_holding_ratio'] = holding_group['ts_profit'] / holding_group['ts_asset']

        holding_group['ts_180profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(180).apply(lambda x : x[-1] - x[0])
        holding_group['ts_90profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(90).apply(lambda x : x[-1] - x[0])
        holding_group['ts_60profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(60).apply(lambda x : x[-1] - x[0])
        holding_group['ts_30profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(30).apply(lambda x : x[-1] - x[0])
        holding_group['ts_15profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(15).apply(lambda x : x[-1] - x[0])
        holding_group['ts_7profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(7).apply(lambda x : x[-1] - x[0])

        #print holding_group.tail()
        #print len(holding_group.columns)

        feat = holding_group

        feat['risk'] = np.nan
        #print feat.head()

        #feat['180buy_num'] = np.nan
        #feat['90buy_num'] = np.nan
        #feat['60buy_num'] = np.nan
        #feat['30buy_num'] = np.nan
        #feat['15buy_num'] = np.nan
        #feat['7buy_num'] = np.nan


        #order_group = order_group.reset_index()
        enc = preprocessing.OneHotEncoder()
        trade_type = order_group['ts_trade_type'].ravel()
        trade_type = trade_type.reshape(len(trade_type), 1)
        enc.fit(trade_type)
        trade_type_cols = ['trade_type' + str(c) for c in enc.active_features_]
        trade_type_df = pd.DataFrame(enc.transform(trade_type).toarray(), columns = trade_type_cols, index = order_group.index)
        trade_type_df = trade_type_df.groupby(trade_type_df.index).first()
        feat = pd.concat([feat, trade_type_df], axis = 1, join_axes = [feat.index])
        feat[trade_type_cols] = feat[trade_type_cols].fillna(0.0)


        for col in trade_type_cols:
            feat[col + '_180_sum'] = feat[col].rolling(180).sum()
            feat[col + '_90_sum'] = feat[col].rolling(90).sum()
            feat[col + '_60_sum'] = feat[col].rolling(60).sum()
            feat[col + '_30_sum'] = feat[col].rolling(30).sum()
            feat[col + '_15_sum'] = feat[col].rolling(15).sum()
            feat[col + '_7_sum'] = feat[col].rolling(7).sum()

        #print feat.tail()
        #print trade_type_df.head()
        #print order_group
        #print trade_type_df
        #order_group = order_group.set_index(['ts_placed_date','ts_trade_type'])
        #trade_type_df = order_group.groupby(['ts_placed_date','ts_trade_type']).count()
        #trade_type_df = trade_type_df.set_index(['ts_placed_date'])
        #print trade_type_df

        #print order_group.head()
        trade_type3_df = order_group[order_group['ts_trade_type'] == 3]
        trade_type3_df = trade_type3_df[['ts_placed_amount']]
        trade_type3_df = trade_type3_df.groupby(trade_type3_df.index).sum()
        print trade_type3_df.head()

        for date in dates:
            if date in order_group.index:
                order_tmp = order_group.loc[date]
                if not isinstance(order_tmp, pd.DataFrame):
                    order_tmp = order_tmp.to_frame().T
                order_tmp = order_tmp.drop_duplicates()
                for i in range(0 ,len(order_tmp)):
                    record = order_tmp.iloc[i]
                    trade_type = record['ts_trade_type']
                    ts_risk = record['ts_risk']
                    if 3 == trade_type:
                        feat.loc[date, 'risk'] = ts_risk



        feat['risk'] = feat['risk'].fillna(method = 'pad')


        feat['risk_180'] = feat['risk'].rolling(180).mean()
        feat['risk_90'] = feat['risk'].rolling(90).mean()
        feat['risk_60'] = feat['risk'].rolling(60).mean()
        feat['risk_30'] = feat['risk'].rolling(30).mean()
        feat['risk_15'] = feat['risk'].rolling(15).mean()
        feat['risk_7'] = feat['risk'].rolling(7).mean()


        #holding_group.index.name = 'date'
        #order_group.index.name = 'date'
        #print holding_group.index
        #print order_group.index


        #print holding_order_df.head()
        feats.append(feat)

    feat_df = pd.concat(feats, axis = 1)

    #print len(feat_df)

    feat_df.to_csv('feat.csv')


@user.command()
@click.option('--featurefile', 'optfeaturefile', default=True, help=u'feature file path')
@click.pass_context
def xgboost(ctx):
    pass
