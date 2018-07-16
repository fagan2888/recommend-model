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
import multiprocessing
from multiprocessing import Manager

from datetime import datetime, timedelta
from dateutil.parser import parse
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from util import xdict
from util.xdebug import dd
from db import database, trade
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from ipdb import set_trace
import json


import traceback, code

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def user(ctx):

    '''user analysis
    '''
    pass



@user.command()
@click.pass_context
def ts_order(ctx):

    engine = database.connection('trade')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(trade.ts_order.ts_uid, trade.ts_order.ts_trade_type, trade.ts_order.ts_trade_status, trade.ts_order.ts_placed_date, trade.ts_order.ts_risk, trade.ts_order.ts_placed_percent, trade.ts_order.ts_placed_amount).filter(trade.ts_order.ts_trade_type.in_([3,4,5,6])).statement
    ts_order = pd.read_sql(sql, session.bind, index_col = ['ts_uid','ts_placed_date', 'ts_trade_type'], parse_dates = ['ts_placed_date'])
    session.commit()
    session.close()

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local')
            .set("spark.executor.memory", "50G")
            .set('spark.driver.memory', '50G')
            .set('spark.driver.maxResultSize', '50G'))
    sc = SparkContext(conf = spark_conf)

    ts_order_rdd = sc.parallelize(ts_order.groupby(level = [0]), 100)

    def buy_date_info(x):
        v = x[1]
        v['ts_buy_date'] = min(v.index.get_level_values(1))
        return v

    ts_order = pd.concat(ts_order_rdd.map(buy_date_info).collect(), axis = 0)
    print(ts_order.tail())

    ts_order_data = {}
    for k, v in ts_order.groupby(level = [0,1,2]):
        ts_placed_amount = np.sum(v.ts_placed_amount.ravel())
        ts_placed_percent = 1.0
        for p in v.ts_placed_percent.ravel():
            ts_placed_percent = ts_placed_percent * (1 - p)
        ts_placed_percent = 1.0 - ts_placed_percent
        v = v.iloc[-1].copy()
        v.ts_placed_amount = ts_placed_amount
        v.ts_placed_percent = ts_placed_percent
        ts_order_data[k] = v
        #print(v)

    ts_order = pd.DataFrame(ts_order_data).T
    ts_order.index.names = ['ts_uid','ts_placed_date', 'ts_trade_type']
    #print(ts_order)
    ts_order.to_csv('tmp/ts_order.csv')
    #ts_order = pd.read_csv('tmp/ts_order.csv', index_col = ['ts_uid'], parse_dates = ['ts_placed_date'])
    #for k, v in ts_order.groupby(level = [0]):
    #    v = v.reset_index().set_index(['ts_placed_date']).sort_index()
    #    if 4 in set(v.ts_trade_type.ravel()):
    #    #print(k, v.ts_trade_type)
    #        print(v)

@user.command()
@click.pass_context
def ts_holding_nav(ctx):

    #engine = database.connection('trade')
    #Session = sessionmaker(bind=engine)
    #session = Session()
    #sql = session.query(trade.ts_holding_nav.ts_uid, trade.ts_holding_nav.ts_portfolio_id, trade.ts_holding_nav.ts_date, trade.ts_holding_nav.ts_nav, trade.ts_holding_nav.ts_asset, trade.ts_holding_nav.ts_profit, trade.ts_holding_nav.ts_processing_asset).statement
    #ts_holding_nav = pd.read_sql(sql, session.bind, index_col = ['ts_uid'], parse_dates = ['ts_date'])
    #ts_holding_nav.to_csv('tmp/ts_holding_nav.csv')
    #session.commit()
    #session.close()

    ts_holding_nav = pd.read_csv('tmp/ts_holding_nav.csv', index_col = ['ts_uid','ts_date'], parse_dates = ['ts_date'])

    #for k, v in ts_holding_nav.groupby(level = [0]):
    #    v = v.reset_index().set_index(['ts_date']).sort_index()
    #    print(v.iloc[0].ts_asset)


    spark_conf = (SparkConf().setAppName('order holding').setMaster('local')
            .set("spark.executor.memory", "50G")
            .set('spark.driver.memory', '50G')
            .set('spark.driver.maxResultSize', '50G'))

    sc = SparkContext(conf = spark_conf)

    ts_holding_nav_rdd = sc.parallelize(ts_holding_nav.groupby(level = [0]), 100)

    def holding_derive_info(x):
        v = x[1]
        v = v.copy().reset_index().set_index(['ts_date']).sort_index()
        v['ts_buy_date'] = min(v.index)
        v['buy_amount'] = v.iloc[0].ts_asset
        v['ts_total_asset'] = v.ts_asset + v.ts_processing_asset
        v['ts_inc'] = v.ts_nav.pct_change().fillna(0.0)
        v['ts_std'] = v.ts_inc.rolling(len(v.ts_inc), min_periods = 1).std().fillna(0.0)
        v['ts_profit_diff'] = v.ts_profit.diff().fillna(0.0)
        v = v.shift(1).iloc[1:]
        return v

    ts_holding_nav = pd.concat(ts_holding_nav_rdd.map(holding_derive_info).collect(), axis = 0)
    print(ts_holding_nav.tail())
    ts_holding_nav.to_csv('tmp/ts_holding_nav_derive_info.csv')


@user.command()
@click.pass_context
def order_holding(ctx):


    #spark = SparkSession.builder.master("local").appName("order holding").getOrCreate()
    #df = spark.read.csv("tmp/ts_order.csv", header = True)
    #df.groupby('ts_uid')


    ts_order = pd.read_csv('tmp/ts_order.csv', index_col = ['ts_uid', 'ts_placed_date'], parse_dates = ['ts_placed_date'])
    ts_holding_nav = pd.read_csv('tmp/ts_holding_nav_derive_info.csv', index_col = ['ts_uid', 'ts_date'], parse_dates = ['ts_date'])
    #ts_hodling_nav = ts_holding_nav[['ts_buy_date']]

    ts_order = ts_order.drop_duplicates()
    ts_order = ts_order[ts_order.ts_placed_amount >= 1000.0]
    #ts_order = ts_order[ts_order.ts_trade_type == 4]

    redeem_order = ts_order[(ts_order.ts_trade_type == 4) & (ts_order.ts_trade_status == 6)]
    redeem_order = redeem_order[redeem_order.ts_placed_percent >= 0.75]
    #buy_order = ts_order[(ts_order.ts_trade_type == 3) & (ts_order.ts_trade_status == 6)]
    #adjust_order = ts_order[(ts_order.ts_trade_type == 6) & (ts_order.ts_trade_status == 6)]

    redeem_holding_order_df = pd.concat([redeem_order, ts_holding_nav], axis = 1, join_axes = [ts_holding_nav.index])[['ts_risk', 'ts_buy_date']]
    #buy_holding_order_df = pd.concat([buy_order, ts_holding_nav], axis = 1, join_axes = [buy_order.index])
    #adjust_holding_order_df = pd.concat([adjust_order, ts_holding_nav], axis = 1, join_axes = [adjust_order.index])

    print(redeem_holding_order_df.tail())
    #print(buy_holding_nav_df.tail())
    #print(adjust_holding_nav_df.tail())
    #df.to_csv('tmp/user_order_holding.csv')

    for k , v in redeem_holding_order_df.groupby(level = [0]):
        v = v.loc[k]
        #print(v)
        #print(v.ts_risk, v.ts_buy_date, k[1])


    #df  = pd.read_csv('tmp/user_order_holding.csv', index_col = ['ts_uid', 'ts_date'], parse_dates = ['ts_date'])

    #df['total_asset'] = df['ts_asset'] + df['ts_processing_asset']
    #df = df.drop(columns = ['ts_asset', 'ts_processing_asset', 'ts_portfolio_id'])

    #user_risk_loss = {}
    #user_risk_drawdown = {}
    #user_risk_profit_drawdown = {}

    #for k, v in df.groupby(level = [0]):

    #    v = v.reset_index().set_index(['ts_date']).sort_index()

    #    v['max_profit'] = v.ts_profit.cummax()
    #    v['max_nav_drawdown'] = 1.0 - v.ts_nav / v.ts_nav.cummax()
    #    v['max_profit_drawdown'] = 1.0 - v.ts_profit / v.ts_profit.cummax()

    #    v = v[(v.ts_trade_type == 4) & (v.ts_placed_percent >= 0.75)]

    #    if len(v) > 0:
    #        v = v[v.total_asset == max(v.total_asset)].iloc[0]
    #        loss = v.ts_profit / v.total_asset
    #        nav_drawdown = v.max_nav_drawdown
    #        profit_drawdown = v.max_profit_drawdown
    #        risk_level = v.ts_risk

    #        losses = user_risk_loss.setdefault(risk_level, [])
    #        losses.append(loss)
    #        drawdowns = user_risk_drawdown.setdefault(risk_level, [])
    #        drawdowns.append(nav_drawdown)
    #        profit_drawdowns = user_risk_profit_drawdown.setdefault(risk_level, [])
    #        profit_drawdowns.append(profit_drawdown)


    #for risk in user_risk_loss.keys():
    #    print(risk, np.mean(user_risk_loss[risk]), np.mean(user_risk_drawdown[risk]), np.mean(user_risk_profit_drawdown[risk]))

@user.command()
@click.pass_context
def user_risk_match(ctx):

    ts_order = pd.read_csv('tmp/ts_order.csv', index_col = ['ts_uid', 'ts_placed_date'], parse_dates = ['ts_placed_date', 'ts_buy_date'])

    redeem_order = ts_order[(ts_order.ts_trade_type == 4) & (ts_order.ts_trade_status == 6)]
    redeem_order = redeem_order[redeem_order.ts_placed_percent >= 0.75]
    redeem_order = redeem_order[redeem_order.ts_placed_amount >= 1000.0]
    #redeem_order = redeem_order.groupby(level = [0, 1]).last()
    #print(redeem_order.tail())

    buy_order = pd.read_csv('tmp/ts_order.csv', index_col = ['ts_uid'], parse_dates = ['ts_placed_date', 'ts_buy_date'])
    buy_order = buy_order[(buy_order.ts_trade_type == 3) & (buy_order.ts_trade_status == 6)]

    risk_positive_negative = {}

    for k,v in redeem_order.groupby(level=[0, 1]):
        v = v.loc[k]
        buy_date = v.ts_buy_date
        buy_date_start = buy_date - timedelta(7)
        buy_date_end = buy_date + timedelta(7)
        risk = v.ts_risk
        redeem_date = k[1]
        redeem_date_start = redeem_date - timedelta(7)
        redeem_date_end = redeem_date + timedelta(7)
        redeem_uid = k[0]
        buy_user = buy_order[(buy_order.ts_risk == risk) &
                (buy_order.ts_buy_date >= buy_date_start) & (buy_order.ts_buy_date <= buy_date_end) &
                (buy_order.ts_placed_date >= redeem_date_start) & (buy_order.ts_placed_date <= redeem_date_end)]
        if len(buy_user) > 0:
            positive_negative = risk_positive_negative.setdefault(risk, {})
            positive = positive_negative.setdefault('positive', set())
            negative = positive_negative.setdefault('negative', set())
            positive.add(int(redeem_uid))
            for buy_uid in buy_user.index:
                negative.add(int(buy_uid))

    #print(risk_positive_negative)
    #df = pd.DataFrame(risk_positive_negative).T
    #df.to_csv('tmp/risk_positive_negative.csv')

    datas = {}
    for risk in risk_positive_negative.keys():
        positive_negative = risk_positive_negative[risk]
        print(list(positive_negative['positive']))
        datas[risk] = [json.dumps(list(positive_negative['positive'])), json.dumps(list(positive_negative['negative']))]
        #print(risk, len(positive_negative['positive']), len(positive_negative['negative']))

    df = pd.DataFrame(datas).T
    df.index.name = 'risk'
    df.columns = ['positive', 'negative']
    print(df.tail())
    df.to_csv('tmp/risk_positive_negative.csv')


