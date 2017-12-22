#coding=utf8


import string
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import random
import numpy as np
import os
import time
import logging
import re
import util_numpy as npu
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Pool as ThreadPool
import functools
import dask.dataframe as dd
import gc


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy import MetaData, Table, select, func, literal_column
from tabulate import tabulate
from db import database, base_exchange_rate_index, base_ra_index
from util import xdict
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


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

    db_tongji_uri =  'mysql://root:Mofang123@127.0.0.1/tongji?charset=utf8&use_unicode=0'
    db = create_engine(db_tongji_uri)
    metadata = MetaData(bind=db)

    t = Table('log_raw_apps', metadata, autoload=True)
    columns = [
        t.c.lr_uid,
        t.c.lr_date,
        t.c.lr_time,
        t.c.lr_page,
        t.c.lr_ctrl,
        t.c.lr_oid,
        t.c.lr_ref,
        t.c.lr_ev,
        t.c.lr_ts,
    ]

    uid_d = 1000000154
    uid_u = 1899999999

    s = select(columns).where(t.c.lr_uid >= uid_d).where(t.c.lr_uid <= uid_u).where(t.c.lr_date >= '2016-08-01')
    df = pd.read_sql(s, db)

    print df.head()
    df.to_csv('user_prediction/app_log.csv')

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
    uid_u = 1899999999

    s = select(columns).where(t.c.ts_uid >= uid_d).where(t.c.ts_uid <= uid_u)
    #s = select(columns).where(t.c.ts_uid >= uid_d).where(t.c.ts_uid <= uid_u).where(t.c.ts_date < (datetime.now() - timedelta(1)).strftime('%Y-%m-%d'))
    #s = select(columns).where(t.c.ts_uid >= uid_d).where(t.c.ts_uid <= uid_u).where(t.c.ts_date < (datetime.now() - timedelta(7)).strftime('%Y-%m-%d'))
    df = pd.read_sql(s, db)

    print df.tail()
    df.to_csv('./user_prediction/ts_holding_nav.csv')

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
    uid_u = 1899999999

    s = select(columns).where(t.c.ts_uid >= uid_d).where(t.c.ts_uid <= uid_u)
    #s = select(columns).where(t.c.ts_uid >= uid_d).where(t.c.ts_uid <= uid_u).where(t.c.ts_placed_date < (datetime.now() - timedelta(1)).strftime('%Y-%m-%d'))
    #s = select(columns).where(t.c.ts_uid >= uid_d).where(t.c.ts_uid <= uid_u).where(t.c.ts_placed_date < (datetime.now() - timedelta(7)).strftime('%Y-%m-%d'))
    df = pd.read_sql(s, db)

    print df.tail()
    df.to_csv('./user_prediction/ts_order.csv')

    return df



def m_analysis_feature(queue, uid_set, holding_df, order_df, log_df):

    uids = set()
    for uid, holding_group in holding_df.groupby(holding_df.index):
        if uid not in set(uid_set):
            continue
        uids.add(uid)
        print uid, len(uids)
        #if len(uids) >= 100:
        #    break

        #print holding_group.head()
        holding_group = holding_group.reset_index()
        holding_group = holding_group.set_index(['ts_date'])
        start_date = holding_group.index[0]
        end_date = holding_group.index[-1]
        holding_group = holding_group.reindex(pd.date_range(start_date , end_date))
        holding_group = holding_group.fillna(method = 'pad')


        if uid not in set(order_df.index):
            continue
        order_group = order_df.loc[uid]

        if not isinstance(order_group, pd.DataFrame):
            order_group = order_group.to_frame().T
        order_group = order_group.reset_index()
        order_group = order_group.set_index(['ts_placed_date'])
        order_dates = list(order_group.index)
        order_dates.sort()
        order_group = order_group.loc[order_dates]

        start_date = order_dates[0]

        if uid not in set(log_df.index):
            continue
        log_group = log_df.loc[uid]
        if not isinstance(log_group, pd.DataFrame):
            log_group = log_group.to_frame().T
        if len(log_group) <= 1:
            continue
        log_group = log_group.reset_index()
        log_group = log_group.set_index(['lr_date'])
        log_dates = list(log_group.index)
        log_dates.sort()
        log_group = log_group.loc[log_dates]
        log_group = log_group[log_group.index >= start_date]

        #print log_group.columns
        log_group = log_group.drop(['lr_uid' , 'lr_time', 'lr_page', 'lr_ctrl', 'lr_ref', 'lr_ev','lr_ts', 'lr_oid', 'lr_action'], axis = 1)
        log_group = log_group.groupby(log_group.index).sum()
        #print log_group.head()


        holding_group = holding_group[holding_group.index >= start_date]
        end_date = order_dates[-1]
        if 4 in order_group.loc[end_date, 'ts_trade_type'].ravel():
            holding_group = holding_group[holding_group.index < end_date]


        dates = holding_group.index
        holding_group['ts_r'] = holding_group['ts_nav'].pct_change().fillna(0.0)

        holding_group['ts_drawdown'] = 1 - holding_group['ts_nav'] / holding_group['ts_nav'].cummax().astype(np.float16)

        holding_group['ts_drawdown_180diff'] = holding_group['ts_drawdown'].rolling(180).apply(lambda x : x[-1] - x[0]).astype(np.float16)
        holding_group['ts_drawdown_90diff'] = holding_group['ts_drawdown'].rolling(90).apply(lambda x : x[-1] - x[0]).astype(np.float16)
        holding_group['ts_drawdown_60diff'] = holding_group['ts_drawdown'].rolling(60).apply(lambda x : x[-1] - x[0]).astype(np.float16)
        holding_group['ts_drawdown_30diff'] = holding_group['ts_drawdown'].rolling(30).apply(lambda x : x[-1] - x[0]).astype(np.float16)
        holding_group['ts_drawdown_15diff'] = holding_group['ts_drawdown'].rolling(15).apply(lambda x : x[-1] - x[0]).astype(np.float16)
        holding_group['ts_drawdown_7diff']  = holding_group['ts_drawdown'].rolling(7).apply(lambda x : x[-1] - x[0]).astype(np.float16)

        holding_group['ts_r_180sum'] = holding_group['ts_r'].rolling(180).sum().astype(np.float16)
        holding_group['ts_r_90sum'] = holding_group['ts_r'].rolling(90).sum().astype(np.float16)
        holding_group['ts_r_60sum'] = holding_group['ts_r'].rolling(60).sum().astype(np.float16)
        holding_group['ts_r_30sum'] = holding_group['ts_r'].rolling(30).sum().astype(np.float16)
        holding_group['ts_r_15sum'] = holding_group['ts_r'].rolling(15).sum().astype(np.float16)
        holding_group['ts_r_7sum'] = holding_group['ts_r'].rolling(7).sum().astype(np.float16)

        holding_group['ts_r_180std'] = holding_group['ts_r'].rolling(180).std().astype(np.float16)
        holding_group['ts_r_90std'] = holding_group['ts_r'].rolling(90).std().astype(np.float16)
        holding_group['ts_r_60std'] = holding_group['ts_r'].rolling(60).std().astype(np.float16)
        holding_group['ts_r_30std'] = holding_group['ts_r'].rolling(30).std().astype(np.float16)
        holding_group['ts_r_15std'] = holding_group['ts_r'].rolling(15).std().astype(np.float16)
        holding_group['ts_r_7std'] = holding_group['ts_r'].rolling(7).std().astype(np.float16)

        holding_group['ts_share_180_chg'] = holding_group['ts_share'].rolling(180).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_share_90_chg'] = holding_group['ts_share'].rolling(90).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_share_60_chg'] = holding_group['ts_share'].rolling(60).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_share_30_chg'] = holding_group['ts_share'].rolling(30).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_share_15_chg'] = holding_group['ts_share'].rolling(15).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_share_7_chg'] = holding_group['ts_share'].rolling(7).apply(lambda x : x[-1] / x[0]).astype(np.float32)

        holding_group['ts_share_180_diff'] = holding_group['ts_share'].rolling(180).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_share_90_diff'] = holding_group['ts_share'].rolling(90).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_share_60_diff'] = holding_group['ts_share'].rolling(60).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_share_30_diff'] = holding_group['ts_share'].rolling(30).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_share_15_diff'] = holding_group['ts_share'].rolling(15).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_share_7_diff'] = holding_group['ts_share'].rolling(7).apply(lambda x : x[-1] - x[0]).astype(np.float32)

        holding_group['ts_asset_180_diff'] = holding_group['ts_asset'].rolling(180).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_asset_90_diff'] = holding_group['ts_asset'].rolling(90).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_asset_60_diff'] = holding_group['ts_asset'].rolling(60).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_asset_30_diff'] = holding_group['ts_asset'].rolling(30).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_asset_15_diff'] = holding_group['ts_asset'].rolling(15).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_asset_7_diff'] = holding_group['ts_asset'].rolling(7).apply(lambda x : x[-1] - x[0]).astype(np.float32)

        holding_group['ts_asset_180_chg'] = holding_group['ts_asset'].rolling(180).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_asset_90_chg'] = holding_group['ts_asset'].rolling(90).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_asset_60_chg'] = holding_group['ts_asset'].rolling(60).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_asset_30_chg'] = holding_group['ts_asset'].rolling(30).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_asset_15_chg'] = holding_group['ts_asset'].rolling(15).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_asset_7_chg'] = holding_group['ts_asset'].rolling(7).apply(lambda x : x[-1] / x[0]).astype(np.float32)

        #print holding_group.tail(10)
        holding_group['ts_profit_180_chg'] = holding_group['ts_profit'].rolling(180).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_profit_90_chg'] = holding_group['ts_profit'].rolling(90).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_profit_60_chg'] = holding_group['ts_profit'].rolling(60).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_profit_30_chg'] = holding_group['ts_profit'].rolling(30).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_profit_15_chg'] = holding_group['ts_profit'].rolling(15).apply(lambda x : x[-1] / x[0]).astype(np.float32)
        holding_group['ts_profit_7_chg'] = holding_group['ts_profit'].rolling(7).apply(lambda x : x[-1] / x[0]).astype(np.float32)

        holding_group['ts_profit_180_diff'] = holding_group['ts_profit'].rolling(180).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_profit_90_diff'] = holding_group['ts_profit'].rolling(90).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_profit_60_diff'] = holding_group['ts_profit'].rolling(60).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_profit_30_diff'] = holding_group['ts_profit'].rolling(30).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_profit_15_diff'] = holding_group['ts_profit'].rolling(15).apply(lambda x : x[-1] - x[0]).astype(np.float32)
        holding_group['ts_profit_7_diff'] = holding_group['ts_profit'].rolling(7).apply(lambda x : x[-1] - x[0]).astype(np.float32)


        #print holding_group.tail()
        #holding_group['ts_profit_holding_ratio'] = holding_group['ts_profit'] / holding_group['ts_asset']
        #print holding_group[['ts_asset', 'ts_profit', 'ts_processing_asset']]

        #holding_group['ts_180profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(180).apply(lambda x : x[-1] - x[0])
        #holding_group['ts_90profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(90).apply(lambda x : x[-1] - x[0])
        #holding_group['ts_60profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(60).apply(lambda x : x[-1] - x[0])
        #holding_group['ts_30profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(30).apply(lambda x : x[-1] - x[0])
        #holding_group['ts_15profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(15).apply(lambda x : x[-1] - x[0])
        #holding_group['ts_7profit_holding_ratio_diff'] = holding_group['ts_profit_holding_ratio'].rolling(7).apply(lambda x : x[-1] - x[0])

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


        order_group = order_group.reset_index()
        order_group = order_group.set_index(['ts_placed_date', 'ts_placed_time'])
        trade_type_cols = ['trade_type2', 'trade_type3', 'trade_type4', 'trade_type5', 'trade_type6', 'trade_type7', 'trade_type8']
        trade_type_df = order_group[trade_type_cols]
        trade_type_df = trade_type_df.groupby(level = [0, 1]).first()
        #print trade_type_df
        trade_type_df = trade_type_df.reset_index()
        trade_type_df = trade_type_df.drop(['ts_placed_time'], axis = 1)
        trade_type_df = trade_type_df.set_index(['ts_placed_date'])
        trade_type_df = trade_type_df.groupby(trade_type_df.index).sum()
        #print trade_type_df
        feat = pd.concat([feat, trade_type_df], axis = 1, join_axes = [feat.index])
        feat[trade_type_cols] = feat[trade_type_cols].fillna(0.0)

        order_group = order_group.reset_index()
        order_group = order_group.set_index(['ts_placed_date'])

        for col in trade_type_cols:
            feat[col + '_180_sum'] = feat[col].rolling(180).sum().astype(np.float16)
            feat[col + '_90_sum'] = feat[col].rolling(90).sum().astype(np.float16)
            feat[col + '_60_sum'] = feat[col].rolling(60).sum().astype(np.float16)
            feat[col + '_30_sum'] = feat[col].rolling(30).sum().astype(np.float16)
            feat[col + '_15_sum'] = feat[col].rolling(15).sum().astype(np.float16)
            feat[col + '_7_sum'] = feat[col].rolling(7).sum().astype(np.float16)


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


        feat['label'] = 0
        redeem = order_group[order_group['ts_trade_type'] == 4]
        buy    = order_group[order_group['ts_trade_type'] == 3]
        #print feat.index
        #print redeem.index

        if len(buy) > 0:
            for buy_date in buy.index:
                feat_date_list = list(feat.index)
                if buy_date > feat.index[-1]:
                    feat_date_list.append(buy_date)
                elif buy_date < feat.index[0]:
                    buy_date = feat.index[0]
                buy_index = feat_date_list.index(buy_date)
                if buy_index < 7:
                    continue
                start_index = buy_index - 7
                feat.label.iloc[start_index:buy_index] = 1

        if len(redeem) > 0:
            for redeem_date in redeem.index:
                feat_date_list = list(feat.index)
                if redeem_date > feat.index[-1]:
                    feat_date_list.append(redeem_date)
                elif redeem_date < feat.index[0]:
                    redeem_date = feat.index[0]
                redeem_index = feat_date_list.index(redeem_date)
                start_index = max(0, redeem_index - 7)
                feat.label.iloc[start_index:redeem_index] = 2


        feat['risk'] = feat['risk'].fillna(method = 'pad')

        feat['risk_180_mean'] = feat['risk'].rolling(180).mean().astype(np.float16)
        feat['risk_90_mean'] = feat['risk'].rolling(90).mean().astype(np.float16)
        feat['risk_60_mean'] = feat['risk'].rolling(60).mean().astype(np.float16)
        feat['risk_30_mean'] = feat['risk'].rolling(30).mean().astype(np.float16)
        feat['risk_15_mean'] = feat['risk'].rolling(15).mean().astype(np.float16)
        feat['risk_7_mean'] = feat['risk'].rolling(7).mean().astype(np.float16)


        log_group = log_group.reindex(feat.index).fillna(0.0)

        for col in log_group.columns:

            log_group[col + '_180_mean'] = log_group[col].rolling(180).mean().astype(np.float16)
            log_group[col + '_90_mean'] = log_group[col].rolling(90).mean().astype(np.float16)
            log_group[col + '_60_mean'] = log_group[col].rolling(60).mean().astype(np.float16)
            log_group[col + '_30_mean'] = log_group[col].rolling(30).mean().astype(np.float16)
            log_group[col + '_15_mean'] = log_group[col].rolling(15).mean().astype(np.float16)
            log_group[col + '_7_mean'] = log_group[col].rolling(7).mean().astype(np.float16)

            log_group[col + '_180_std'] = log_group[col].rolling(180).std().astype(np.float16)
            log_group[col + '_90_std'] = log_group[col].rolling(90).std().astype(np.float16)
            log_group[col + '_60_std'] = log_group[col].rolling(60).std().astype(np.float16)
            log_group[col + '_30_std'] = log_group[col].rolling(30).std().astype(np.float16)
            log_group[col + '_15_std'] = log_group[col].rolling(15).std().astype(np.float16)
            log_group[col + '_7_std'] = log_group[col].rolling(7).std().astype(np.float16)


        feat = pd.concat([feat, log_group], axis = 1, join_axes = [feat.index])
        #feat.to_csv('user_prediction/feat/' + str(uid) + '.csv')


        #print feat.tail()

        #holding_group.index.name = 'date'
        #order_group.index.name = 'date'
        #print holding_group.index
        #print order_group.index
        #print feat.columns

        queue.put(feat)




@user.command()
#@click.option('--logfile', 'optlogfile', default=True, help=u'logfile path')
@click.pass_context
def analysis_feature(ctx):


    #log_df = pd.read_csv('user_prediction/app_log.csv')
    holding_df = pd.read_csv('user_prediction/ts_holding_nav.csv', parse_dates = ['ts_date'], dtype = {'ts_nav':np.float32, 'ts_share':np.float32, 'ts_asset':np.float32, 'ts_processing_asset':np.float32, 'ts_profit':np.float32})

    order_df = pd.read_csv('user_prediction/ts_order.csv', parse_dates = ['ts_placed_date'], dtype = {'ts_trade_type':np.int16})

    #print holding_df.info(memory_usage='deep')

    #print holding_df.info()
    #sys.exit(0)

    #print order_df.columns
    order_cols = ['ts_uid', 'ts_trade_type', 'ts_trade_status', 'ts_placed_date', 'ts_placed_time',
                            'ts_placed_amount','ts_placed_percent', 'ts_acked_amount', 'ts_risk']
    order_df = order_df[order_cols]
    order_df = order_df.dropna()


    log_cols = ['lr_uid', 'lr_date' , 'lr_time' , 'lr_page' , 'lr_oid' , 'lr_ctrl' , 'lr_ref', 'lr_ev', 'lr_ts']
    log_df = pd.read_csv('user_prediction/app_log.csv', parse_dates = ['lr_date'], dtype = {'lr_page':np.int16, 'lr_ctrl':np.int16, 'lr_ref':np.int16, 'lr_ev':np.int16, 'lr_oid':np.int32})
    log_df = log_df[log_cols]

    log_df = log_df[log_df['lr_page'] > 2001]
    log_df = log_df[log_df['lr_ref'] > 2001]

    log_df['lr_ctrl'] = log_df['lr_ctrl'].replace(-1, 200)
    #log_df = log_df.set_index(['lr_uid'])

    log_df = log_df[log_df['lr_uid'].isin(set(order_df['ts_uid']))]

    recent_log_df = log_df[log_df['lr_date'] >= '2017-11-01']

    pages = set(recent_log_df['lr_page'])
    ctrls = set(recent_log_df['lr_ctrl'])
    refs  = set(recent_log_df['lr_ref'])

    log_df = log_df[log_df['lr_page'].isin(pages)]
    log_df = log_df[log_df['lr_ctrl'].isin(ctrls)]
    log_df = log_df[log_df['lr_ref'].isin(refs)]

    #log_df = log_df.set_index('lr_uid')




    lb = preprocessing.LabelBinarizer()
    trade_type = order_df['ts_trade_type'].ravel()
    lb.fit(trade_type)
    trade_type_cols = ['trade_type' + str(c) for c in lb.classes_]
    trade_type_df = pd.DataFrame(np.float16(lb.transform(trade_type)), columns = trade_type_cols, index = order_df.index)

    order_df = pd.concat([order_df, trade_type_df], axis = 1, join_axes = [order_df.index])
    #print order_df.head()
    #print order_df.tail()
    #holding_group[trade_type_cols] = holding_group[trade_type_cols].fillna(0.0)


    order_df = order_df.set_index(['ts_uid'])
    #print 1000262865 in set(order_df.index)
    #order_df = order_df[order_df['ts_trade_status'] > 0]
    order_df = order_df[order_df['ts_trade_type'] < 7]
    #print 1000262865 in set(order_df.index)
    #order_df = order_df[order_df['ts_trade_type'] == 4]
    #order_df = order_df['ts_invest_plan_id']
    #print order_df


    holding_cols = ['ts_uid', 'ts_date', 'ts_nav', 'ts_share', 'ts_asset', 'ts_processing_asset', 'ts_profit']
    holding_df = holding_df[holding_cols]
    holding_df = holding_df.set_index(['ts_uid'])
    #print holding_df.head()

    #holding_order_df = pd.concat([holding_df, order_df], axis = 1, join_axes = [holding_df.index])
    #print holding_order_df

    #print set(log_df['lr_page'])
    #print set(log_df['lr_ctrl'])
    #print set(log_df['lr_ref'])
    #print set(log_df['lr_ev'])


    log_df['lr_action'] = 'unknow'
    log_df = log_ctrl(log_df)

    lb = preprocessing.LabelBinarizer()
    page = log_df['lr_page'].ravel()
    lb.fit(page)
    page_cols = ['lr_page' + str(c) for c in lb.classes_]
    page_df = pd.DataFrame(lb.transform(page), columns = page_cols, index = log_df.index)
    #print type(page_df.iloc[0,0])

    lb = preprocessing.LabelBinarizer()
    action = log_df['lr_action'].ravel()
    lb.fit(action)
    action_cols = ['lr_action' + str(c) for c in lb.classes_]
    action_df = pd.DataFrame(np.float16(lb.transform(action)), columns = action_cols, index = log_df.index)
    #print ref_df.head()

    lb = preprocessing.LabelBinarizer()
    ref = log_df['lr_ref'].ravel()
    lb.fit(ref)
    ref_cols = ['lr_ref' + str(c) for c in lb.classes_]
    ref_df = pd.DataFrame(np.float16(lb.transform(ref)), columns = ref_cols, index = log_df.index)

    log_df = pd.concat([log_df, page_df, ref_df, action_df], axis = 1, join_axes = [log_df.index])
    log_df = log_df.set_index(['lr_uid'])

    print 'LOG ORDER HOLDING PAGE ctrl ref DONE'
    #time.sleep(10)

    count = multiprocessing.cpu_count() / 2
    #count = 1
    uids = list(set(holding_df.index))
    process_uid_indexs = [[] for i in range(0, count)]
    for i in range(0, len(uids)):
       process_uid_indexs[i % count].append(uids[i])


    manager = Manager()
    q = manager.Queue()
    processes = []
    for indexs in process_uid_indexs:
        p = multiprocessing.Process(target = m_analysis_feature, args = (q, indexs, holding_df, order_df, log_df))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


    feats = []
    for m in range(0, q.qsize()):
        record = q.get(m)
        feats.append(record)

    feat_df = pd.concat(feats, axis = 0)
    feat_df.index.name = 'ts_date'

    #print len(feat_df)

    feat_df.to_csv('./user_prediction/feat.csv')


def log_ctrl(log_df):


    log_df.lr_action[(log_df.lr_ctrl == 4) & (log_df.lr_ref == 2013)] = 'ctrl=4&ref=2013'
    log_df.lr_action[(log_df.lr_ctrl == 99) & (log_df.lr_ref == 2110)] = 'ctrl=99&ref=2110'
    log_df.lr_action[(log_df.lr_ctrl == 6) & (log_df.lr_page == 2013)] = 'ctrl=6&pageid=2013'
    log_df.lr_action[(log_df.lr_ctrl == 99) & (log_df.lr_ref == 2162)] = 'ctrl=99&ref=2162'
    log_df.lr_action[(log_df.lr_ctrl == 7) & (log_df.lr_ref == 2013)] = 'ctrl=7&ref=2013'
    log_df.lr_action[(log_df.lr_ctrl == 8) & (log_df.lr_ref == 2013)] = 'ctrl=8&ref=2013'
    log_df.lr_action[(log_df.lr_ctrl == 9) & (log_df.lr_ref == 2013)] = 'ctrl=9&ref=2013'
    log_df.lr_action[(log_df.lr_ctrl == 10) & (log_df.lr_ref == 2013)] = 'ctrl=10&ref=2013'
    log_df.lr_action[(log_df.lr_ctrl == 11) & (log_df.lr_ref == 2013)] = 'ctrl=11&ref=2013'
    log_df.lr_action[(log_df.lr_ctrl == 12) & (log_df.lr_ref == 2013)] = 'ctrl=12&ref=2013'
    log_df.lr_action[(log_df.lr_ctrl == 13) & (log_df.lr_ref == 2013)] = 'ctrl=13&ref=2013'
    log_df.lr_action[(log_df.lr_ctrl == 2) & (log_df.lr_ref == 2006)] = 'ctrl=2&ref=2006'
    log_df.lr_action[(log_df.lr_oid == 1) & (log_df.lr_ctrl == 2) & (log_df.lr_ref == 2006)] = 'oid=1&ctrl=2&ref=2006'
    log_df.lr_action[(log_df.lr_oid == 2) & (log_df.lr_ctrl == 2) & (log_df.lr_ref == 2006)] = 'oid=2&ctrl=2&ref=2006'
    log_df.lr_action[(log_df.lr_oid == 3) & (log_df.lr_ctrl == 2) & (log_df.lr_ref == 2006)] = 'oid=3&ctrl=2&ref=2006'
    log_df.lr_action[(log_df.lr_oid == 4) & (log_df.lr_ctrl == 2) & (log_df.lr_ref == 2006)] = 'oid=4&ctrl=2&ref=2006'
    log_df.lr_action[(log_df.lr_oid == 5) & (log_df.lr_ctrl == 2) & (log_df.lr_ref == 2006)] = 'oid=5&ctrl=2&ref=2006'
    log_df.lr_action[(log_df.lr_ctrl == 4) & (log_df.lr_page == 2006)] = 'ctrl=4&pageid=2006'
    log_df.lr_action[(log_df.lr_ctrl == 3) & (log_df.lr_ref == 2006)] = 'ctrl=3&ref=2006'
    log_df.lr_action[(log_df.lr_ctrl == 0) & (log_df.lr_ref == 2006)] = 'ctrl=0&ref=2006'
    log_df.lr_action[(log_df.lr_ctrl == 2) & (log_df.lr_ref == 2147)] = 'ctrl=2&ref=2147'


    return log_df


@user.command()
@click.option('--featurefile', 'optfeaturefile', default=True, help=u'feature file path')
@click.pass_context
def xgboost(ctx, optfeaturefile):

    '''
    feat_df = pd.read_hdf(optfeaturefile.strip(), mode = 'r')
    feat_df = feat_df.drop(['ts_uid'], axis = 1)

    train_df = feat_df[feat_df.index <= '2017-10-30']
    val_df  = feat_df[(feat_df.index > '2017-11-07') & (feat_df.index <= '2017-12-01')]
    test_df  = feat_df[feat_df.index > '2017-12-01']

    train_df = train_df.reset_index()
    negative_train_df = train_df[train_df.label == 0]
    positive_train_df = train_df[(train_df.label == 1) | (train_df.label == 2)]
    positive_index = positive_train_df.index.ravel()
    negative_index = negative_train_df.index.ravel()
    random.shuffle(negative_index)
    negative_index = negative_index[0: len(negative_index) / 5]
    train_index = np.append(negative_index, positive_index)
    random.shuffle(train_index)
    train_df = train_df.iloc[train_index]
    train_df = train_df.set_index(['ts_date'])


    train_y = train_df['label'].values
    train_X = train_df.drop(['label'], axis = 1).values
    test_y = test_df['label'].values
    test_X = test_df.drop(['label'], axis = 1).values
    val_y = val_df['label'].values
    val_X = val_df.drop(['label'], axis = 1).values

    #train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.2, random_state = 13243)

    xg_train = xgb.DMatrix(train_X, label=train_y, missing = np.nan)
    xg_val = xgb.DMatrix(val_X, label=val_y, missing = np.nan)
    xg_test = xgb.DMatrix(test_X, label=test_y, missing = np.nan)

    xg_train.save_binary('./user_prediction/train.xgb')
    xg_val.save_binary('./user_prediction/val.xgb')
    xg_test.save_binary('./user_prediction/test.xgb')
    sys.exit(0)
    '''


    xg_train = xgb.DMatrix('./user_prediction/train.xgb')
    xg_val = xgb.DMatrix('./user_prediction/val.xgb')
    xg_test = xgb.DMatrix('./user_prediction/test.xgb')

    params = {}
    # use softmax multi-class classification
    params['objective'] = 'multi:softmax'
    #params['booster'] = 'gbtree'
    # scale weight of positive examples
    params['eta'] = 0.1
    params['max_depth'] = 9
    params['silent'] = 1
    params['nthread'] = 32
    params['num_class'] = 3
    params['eval_metric'] = 'mlogloss'
    #params['min_child_weight'] = 3
    params['lambda'] = 1
    #params['gamma'] = 0.1
    #params['subsample'] = 0.5
    #params['colsample_bytree'] = 0.5
    #params['rate_drop'] = 0.2
    #params['skip_drop'] = 0.5
    #params['scale_pos_weight'] = float(np.sum(train_y == 0)) / np.sum(train_y==1)
    #params['seed'] = 103
    #params['tree_method'] = 'gpu_hist'
    #params['gpu_id'] = 0
    params['max_bin'] = 32
    #params['predictor'] = 'cpu_predictor'
    #params['verbose'] = 0


    #watchlist = [(xg_train,'train'), (xg_val,'val')]
    #num_round = 2000
    #clf = xgb.train(params, xg_train, num_round, watchlist, early_stopping_rounds = 50)
    #best_ntree_limit = clf.best_ntree_limit


    #importance = clf.get_fscore()
    #clf.save_model('./user_prediction/xgboost.model')
    clf = xgb.Booster(params)
    clf.load_model('./user_prediction/xgboost.model')

    #train_pre = clf.predict(xg_train, ntree_limit=best_ntree_limit)
    #test_pre = clf.predict(xg_test, ntree_limit=best_ntree_limit)
    train_pre = clf.predict(xg_train, ntree_limit=408)
    test_pre = clf.predict(xg_test, ntree_limit=408)
    print confusion_matrix(xg_train.get_label(), train_pre)
    print classification_report(xg_train.get_label(), train_pre)
    print confusion_matrix(xg_test.get_label(), test_pre)
    print classification_report(xg_test.get_label(), test_pre)


    feat_df = pd.read_hdf(optfeaturefile.strip(), mode = 'r')
    test_df  = feat_df[feat_df.index > '2017-12-01']

    test_user = test_df[['ts_uid', 'label']]
    test_user['predict'] =  test_pre


    print test_user.index[-1]
    rebuy = set(test_user.ts_uid[test_user.label == 1])
    predict_rebuy = set(test_user.ts_uid[test_user.predict == 1])
    print len(rebuy), len(predict_rebuy), len(rebuy & predict_rebuy)

    print 'rebuy presicion : ', 1.0 * len(rebuy & predict_rebuy) / len(predict_rebuy)
    print 'rebuy recall : ', 1.0 * len(rebuy & predict_rebuy) / len(rebuy)

    redeem = set(test_user.ts_uid[test_user.label == 2])
    predict_redeem = set(test_user.ts_uid[test_user.predict == 2])
    print len(redeem), len(predict_redeem), len(redeem & predict_redeem)

    print 'redeem presicion : ', 1.0 * len(redeem & predict_redeem) / len(predict_redeem)
    print 'redeem recall : ', 1.0 * len(redeem & predict_redeem) / len(redeem)

    return



@user.command()
@click.option('--featurefile', 'optfeaturefile', default=True, help=u'feature file path')
@click.pass_context
def lightgbm(ctx, optfeaturefile):

    feat_df = pd.read_csv(optfeaturefile.strip(), index_col = ['ts_date'], parse_dates = ['ts_date'], nrows = 10)
    dtype = {}
    drop_cols = []
    for col in feat_df.columns:
        if col.startswith('lr_page'):
            dtype[col] = np.float16
        elif col.startswith('ts_drawdown'):
            dtype[col] = np.float16
        elif col.startswith('ts_r'):
            dtype[col] = np.float16
        elif col.startswith('ts_share'):
            dtype[col] = np.float32
        elif col.startswith('ts_asset'):
            dtype[col] = np.float32
        elif col.startswith('ts_profit'):
            dtype[col] = np.float32
        elif col.startswith('trade_type'):
            dtype[col] = np.float16
        elif col.startswith('risk'):
            dtype[col] = np.float16
        elif col.startswith('lr_action'):
            dtype[col] = np.float16
        elif col.startswith('lr_ref'):
            dtype[col] = np.float16
    feat_df = pd.read_csv(optfeaturefile.strip(), index_col = ['ts_date'], parse_dates = ['ts_date'], dtype = dtype)
    #feat_df = feat_df.drop(drop_cols, axis = 1)
    #print feat_df.info()
    #feat_df.loc[feat_df.label == 1, 'label'] = 0
    #feat_df.loc[feat_df.label == 2, 'label'] = 1
    feat_df = feat_df.drop(['ts_uid'], axis = 1)


    train_df = feat_df[feat_df.index <= '2017-11-30']
    test_df  = feat_df[feat_df.index > '2017-12-07']

    train_df = train_df.reset_index()
    negative_train_df = train_df[train_df.label == 0]
    positive_train_df = train_df[(train_df.label == 1) | (train_df.label == 2)]
    positive_index = positive_train_df.index.ravel()
    negative_index = negative_train_df.index.ravel()
    random.shuffle(negative_index)
    negative_index = negative_index[0: len(negative_index) / 10]
    train_index = np.append(negative_index, positive_index)
    random.shuffle(train_index)
    train_df = train_df.iloc[train_index]
    train_df = train_df.set_index(['ts_date'])

    train_y = train_df['label']
    train_X = train_df.drop(['label'], axis = 1)
    test_y = test_df['label']
    test_X = test_df.drop(['label'], axis = 1)

    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.2, random_state = 13243)

    lgb_train = lgb.Dataset(train_X, train_y, free_raw_data=False)
    lgb_val = lgb.Dataset(val_X, val_y, reference=lgb_train, free_raw_data=False)

    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class' : 3,
            'metric': {'multi_logloss'},
            'learning_rate': 0.1,
            #'num_leaves': 112,
            #'max_depth': 9,
            #'min_data_in_leaf': 82,
            #'min_sum_hessian_in_leaf': 1e-3,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            #'bagging_freq': 5,
            #'min_gain_to_split':0,
            #'lambda_l1': 100,
            #'lambda_l2': 1,
            #'max_bin': 255,
            #'drop_rate': 0.2,
            #'skip_drop': 0.5,
            #'max_drop': 50,
            'nthread': 32,
            'verbose': -1,
            #'is_unbalance' : True,
            #'scale_pos_weight':weight
            'device' : 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id' : 0,
            "max_bin" : 63,
            'data_random_seed': 963,
            }


    gbm = lgb.train(params, lgb_train, valid_sets=[lgb_val], num_boost_round=100000,
                    early_stopping_rounds=200,  categorical_feature=['risk'], verbose_eval=True)
    #print cvresult
    #best_round = len(cvresult['multi_logloss-mean'])
    #best_round = gbm.best_iteration_
    #print 'best_num_boost_round: %d' % best_round
    #gbm = lgb.train(params, lgb_train, num_boost_round=best_round, verbose_eval=False)
    train_pre = gbm.predict(train_X)
    test_pre = gbm.predict(test_X)
    train_pre = np.argmax(train_pre, axis = 1)
    test_pre = np.argmax(test_pre, axis = 1)
    #print train_pre
    #print "\nModel Report"
    #print "AUC Score(Train): %f" % roc_auc_score(train_y, train_pre)
    #print "AUC Score(Test) : %f" % roc_auc_score(test_y, test_pre)


    #train_pre[train_pre > 0.5] = 1
    #train_pre[train_pre <= 0.5] = 0
    print confusion_matrix(train_y, train_pre)
    #test_pre[test_pre > 0.5] = 1
    #test_pre[test_pre <= 0.5] = 0
    print confusion_matrix(test_y, test_pre)
    print classification_report(test_y, test_pre)

    #print('Feature names:', gbm.feature_name())
    #print('Feature importances:', list(gbm.feature_importance()))
    #print len(test_y[test_y == 1]), len(test_y)
    #print len(test_pre[test_pre == 1]), len(test_pre)


    feature_importance_df = pd.DataFrame(gbm.feature_importance(), index = gbm.feature_name(), columns = ['value'])
    #print feature_importance_df
    feature_importance_df.to_csv('feature_importance.csv')


    '''
    uids = set(test_df.ts_uid.ravel())
    redeem_uids = set()
    predict_redeem_uids = set()
    test_y = test_y.ravel()
    for i in range(0, len(test_y))
        if test_y[i] == 1:
    '''
    #for i in range(0 ,len(test_y)):

    gbm.save_model('lightgbm.txt')

    return gbm



@user.command()
@click.option('--featurefile', 'optfeaturefile', default=True, help=u'feature file path')
@click.option('--modelfile', 'optmodelfile', default=True, help=u'model file path')
@click.pass_context
def lightgbm_test(ctx, optfeaturefile, optmodelfile):


    feat_df = pd.read_csv(optfeaturefile.strip(), index_col = ['ts_date'], parse_dates = ['ts_date'])
    #feat_df.loc[feat_df.label == 1, 'label'] = 0
    #feat_df.loc[feat_df.label == 2, 'label'] = 1

    #feat_df = feat_df.drop(['ts_uid'], axis = 1)
    test_df  = feat_df[feat_df.index >= '2017-12-12']
    test_df  = test_df[test_df.index < '2017-12-13']
    print len(test_df)
    #print test_df

    #print test_df.to_csv('test.csv')
    uids = test_df['ts_uid'].ravel()
    test_df = test_df.drop(['ts_uid'], axis = 1)

    test_y = test_df['label'].ravel()
    test_X = test_df.drop(['label'], axis = 1)

    bst = lgb.Booster(model_file=optmodelfile.strip())  #init model
    test_pre = bst.predict(test_X)
    #print len(test_pre)

    redeem_uids = []

    for i in range(0, len(test_pre)):
        if test_pre[i] >= 0.5:
            print uids[i]
            redeem_uids.append(uids[i])

    print redeem_uids




@user.command()
@click.option('--drawdown-ratio', 'optdrawdownratio', default=True, help=u'drawdown ratio')
@click.pass_context
def drawdown_user(ctx, optdrawdownratio):

    holding_df = pd.read_csv('user_prediction/ts_holding_nav.csv', index_col = ['ts_uid'])
    holding_df = holding_df[['ts_date','ts_nav']]
    #print holding_df.tail()
    uids = []
    drawdowns = []
    for uid, group in holding_df.groupby(holding_df.index):
        group = group.reset_index()
        group = group[['ts_date','ts_nav']]
        group = group.set_index(['ts_date'])

        if group.index[-1] < '2017-12-07':
            continue
        group['cummax'] = group['ts_nav'].cummax()
        group['drawdown'] = 1.0 - group['ts_nav'] / group['cummax']
        current_drawdown = group['drawdown'][-1]
        if current_drawdown >= float(optdrawdownratio.strip()):
            print uid, current_drawdown
            uids.append(uid)
            drawdowns.append(current_drawdown)

    drawdown_df = pd.DataFrame(drawdowns, index = uids)
    #print len(uids)
    user_account_info_df = pd.read_csv('user_prediction/user_account_infos.csv', index_col = ['uid'])
    #print user_account_info_df

    drawdown_user_account_info_df = pd.concat([user_account_info_df, drawdown_df], axis = 1, join_axes = [drawdown_df.index])
    #print drawdown_user_account_info_df
    drawdown_user_account_info_df.to_csv('drawdown_user_account_info_df.csv', encoding='gbk')
    #print group.tail()
