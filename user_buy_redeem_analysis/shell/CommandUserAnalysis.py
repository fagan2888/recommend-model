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
from db import database, trade, recommend, tongji, mapi, base_ra_index_nav, base_trade_dates
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pickle
from tempfile import NamedTemporaryFile
import functools
from ipdb import set_trace
from esdata import ESData
import json
import warnings
warnings.filterwarnings("ignore")


import traceback, code

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def analysis(ctx):

    '''user analysis
    '''
    pass


@analysis.command()
@click.pass_context
def user_follow_adjust(ctx):

    ts_order = pd.read_csv('./data/ts_order.csv', index_col = ['ts_order_uid'])
    ts_holding_df = pd.read_csv("data/ts_holding_nav.csv" ,index_col = ['ts_holding_uid', 'ts_date'])

    #print(ts_order.tail())

    risk_drawdown_threshold = {}
    risk_drawdown_threshold[1.0] = 0.15
    risk_drawdown_threshold[0.9] = 0.13
    risk_drawdown_threshold[0.8] = 0.112
    risk_drawdown_threshold[0.7] = 0.094
    risk_drawdown_threshold[0.6] = 0.079
    risk_drawdown_threshold[0.5] = 0.064
    risk_drawdown_threshold[0.4] = 0.052
    risk_drawdown_threshold[0.3] = 0.04
    risk_drawdown_threshold[0.2] = 0.03
    risk_drawdown_threshold[0.1] = 0.02


    not_adjust_user = []
    adjust_user = []
    num = 0
    for uid, holding in ts_holding_df.groupby(level = [0]):
        num = num + 1
        #print(num)
        if uid not in set(ts_order.index):
            continue
        user_order = ts_order.loc[[uid]]
        if user_order.iloc[-1]['46'] == 1.0:
            continue 
        #print(user_order.tail())
        user_risk = user_order.ts_risk.iloc[-1]
        if user_risk == 0.1:
            continue
        nav = holding.ts_nav
        drawdown = 1.0 - nav / nav.cummax() 
        drawdown_threshold = risk_drawdown_threshold[user_risk]
        user_order = user_order[user_order['66'] == 1.0]
        user_order = user_order[user_order.ts_trade_date >= '2018-01-01']
        if drawdown.iloc[-1] >= drawdown_threshold:
            adjust_date = user_order.ts_trade_date.tolist()
            if '2018-10-11' not in adjust_date:
                print(uid, user_risk, drawdown.iloc[-1], '2018-10-11')
                not_adjust_user.append([uid, user_risk, drawdown.iloc[-1], '2018-10-11'])
                continue
            elif '2018-02-08' not in adjust_date:
                print(uid, user_risk, drawdown.iloc[-1], '2018-02-08')
                not_adjust_user.append([uid, user_risk, drawdown.iloc[-1], '2018-02-08'])
                continue
            elif '2018-02-06' not in adjust_date:
                print(uid, user_risk, drawdown.iloc[-1], '2018-02-06')
                not_adjust_user.append([uid, user_risk, drawdown.iloc[-1], '2018-02-06'])
                continue
            elif '2018-02-12' not in adjust_date:
                print(uid, user_risk, drawdown.iloc[-1], '2018-02-12')
                not_adjust_user.append([uid, user_risk, drawdown.iloc[-1], '2018-02-12'])
                continue
            elif '2018-05-15' not in adjust_date:
                print(uid, user_risk, drawdown.iloc[-1], '2018-05-15')
                not_adjust_user.append([uid, user_risk, drawdown.iloc[-1], '2018-05-15'])
                continue
            elif '2018-06-08' not in adjust_date:
                print(uid, user_risk, drawdown.iloc[-1], '2018-06-08')
                not_adjust_user.append([uid, user_risk, drawdown.iloc[-1], '2018-06-08'])
                continue
            else:
                adjust_user.append([uid, user_risk, drawdown.iloc[-1]])
                print(uid, user_risk, drawdown.iloc[-1])
            #print()
        #print(drawdown.tail())
    pd.DataFrame(not_adjust_user).to_csv('./tmp/not_adjust_user.csv');
    pd.DataFrame(adjust_user).to_csv('./tmp/adjust_user.csv');


@analysis.command()
@click.pass_context
def user_buy_redeem_ratio(ctx):

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
                .set("spark.executor.memory", "24G")
                .set('spark.driver.memory', '24G')
                .set('spark.driver.maxResultSize', '24G'))

    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    feature = spark.read.csv("data/all_feature.csv", header = True)
    feature_rdd = feature.rdd.repartition(100)
    trade_dates = base_trade_dates.load_index()

    def user_feature_analysis(trade_dates, x):

            uid = x[0]
            user_feature = pd.DataFrame([v.asDict() for v in x[1]]).fillna(np.nan)
            user_feature['date'] = pd.to_datetime(user_feature.date)
            user_feature = user_feature[user_feature.date.isin(trade_dates)]
            user_feature = user_feature.set_index(['uid', 'date'])
            user_feature = user_feature.sort_index()
            user_feature['36'] = user_feature['36'].astype(float)
            user_feature['46'] = user_feature['46'].astype(float)
            user_feature = user_feature[['36', '46']].fillna(0.0)
            #user_feature = user_feature.rolling(window = 30).sum().dropna()
            user_feature.index = user_feature.index.droplevel(0)
            date_buy_redeem = {}
            for date in user_feature.index:
                tag_36 = max(user_feature.loc[date, '36'].ravel())
                tag_46 = max(user_feature.loc[date, '46'].ravel())
                buy_redeem = date_buy_redeem.setdefault(date, {})
                if tag_36 >= 1.0:
                    buy_num = buy_redeem.setdefault('buy',0)
                    buy_redeem['buy'] = buy_num + 1
                elif tag_46 >= 1.0:
                    redeem_num = buy_redeem.setdefault('redeem',0)
                    buy_redeem['redeem'] = redeem_num + 1
                elif (tag_36 == 0.0) and (tag_46 == 0.0):
                    hold_num = buy_redeem.setdefault('hold',0)
                    buy_redeem['hold'] = hold_num + 1
            return date_buy_redeem
        
        
    def buy_redeem_num_reduce(log1, log2):
        for date in log1.keys():
            log2_date_buy_redeem = log2.setdefault(date, {})
            log1_date_buy_redeem = log1[date]
            for tag in log1_date_buy_redeem:
                log1_tag_num = log1_date_buy_redeem[tag]
                log2_tag_num = log2_date_buy_redeem.setdefault(tag, 0)
                log2_date_buy_redeem[tag] = log1_tag_num + log2_tag_num
        return log2

    feature = feature_rdd.groupBy(lambda x : x.uid).map(functools.partial(user_feature_analysis, trade_dates)).reduce(buy_redeem_num_reduce)
    print('feature num', len(feature))
    spark.stop()


    df = pd.DataFrame(feature).T.fillna(0.0)
    df = df.sort_index(ascending = False)
    df['total'] = df.buy + df.hold + df.redeem
    df['buy_ratio'] = df.buy / df.total
    df['redeem_ratio'] = df.redeem / df.total
    df = df[df.index >= '2017-01-01']
    df.to_csv('tmp/buy_redeem_ratio.csv')



@analysis.command()
@click.pass_context
def user_risk_ratio(ctx):

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
                .set("spark.executor.memory", "24G")
                .set('spark.driver.memory', '24G')
                .set('spark.driver.maxResultSize', '24G'))

    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    feature = spark.read.csv("data/all_feature.csv", header = True)
    feature_rdd = feature.rdd.repartition(100)
    trade_dates = base_trade_dates.load_index()

    def user_feature_analysis(trade_dates, x):

            uid = x[0]
            user_feature = pd.DataFrame([v.asDict() for v in x[1]]).fillna(np.nan)
            user_feature['date'] = pd.to_datetime(user_feature.date)
            user_feature = user_feature[user_feature.date.isin(trade_dates)]
            user_feature = user_feature.sort_index()
            user_feature['36'] = user_feature['36'].astype(float)
            user_feature['46'] = user_feature['46'].astype(float)
            
            user_feature = user_feature[['36', '46' ,'ts_risk', 'ts_profit', 'date']].fillna(0.0)
            user_feature = user_feature.set_index(['date', 'ts_risk'])

            return user_feature


    user_data = feature_rdd.groupBy(lambda x : x.uid).map(functools.partial(user_feature_analysis, trade_dates)).reduce(lambda u1, u2: pd.concat([u1, u2], axis = 0))
    spark.stop()


@analysis.command()
@click.pass_context
def user_rebuy_ratio(ctx):

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
                .set("spark.executor.memory", "24G")
                .set('spark.driver.memory', '24G')
                .set('spark.driver.maxResultSize', '24G'))

    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    feature = spark.read.csv("data/all_feature.csv", header = True)
    feature_rdd = feature.rdd.repartition(100)
    trade_dates = base_trade_dates.load_index()

    def user_feature_analysis(trade_dates, x):

            uid = x[0]
            user_feature = pd.DataFrame([v.asDict() for v in x[1]]).fillna(np.nan)
            user_feature['date'] = pd.to_datetime(user_feature.date)
            user_feature = user_feature[user_feature.date.isin(trade_dates)]
            user_feature = user_feature.sort_index()
            user_feature['36'] = user_feature['36'].astype(float)
            user_feature['46'] = user_feature['46'].astype(float)
            user_feature['ts_placed_amount'] = user_feature['ts_placed_amount'].astype(float)
            
            user_feature = user_feature[['36', 'date','ts_placed_amount']].fillna(0.0)
            user_feature = user_feature.set_index(['date'])
            user_feature = user_feature[user_feature['36'] == 1.0]

            placed_amount = user_feature.ts_placed_amount.ravel()
            #print(np.sum(placed_amount))
            if len(placed_amount) == 0:
                return {}
            else:
                return {uid: np.sum(placed_amount) / placed_amount[0] - 1}


    user_data = feature_rdd.groupBy(lambda x : x.uid).map(functools.partial(user_feature_analysis, trade_dates)).reduce(lambda u1, u2: {**u1, **u2})
    spark.stop()



@analysis.command()
@click.pass_context
def user_redeem_day(ctx):

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
                .set("spark.executor.memory", "24G")
                .set('spark.driver.memory', '24G')
                .set('spark.driver.maxResultSize', '24G'))

    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    feature = spark.read.csv("data/all_feature.csv", header = True)
    feature_rdd = feature.rdd.repartition(100)
    trade_dates = base_trade_dates.load_index()

    def user_feature_analysis(trade_dates, x):

            uid = x[0]
            user_feature = pd.DataFrame([v.asDict() for v in x[1]]).fillna(np.nan)
            user_feature['date'] = pd.to_datetime(user_feature.date)
            user_feature = user_feature[user_feature.date.isin(trade_dates)]
            user_feature = user_feature.sort_index()
            user_feature['36'] = user_feature['36'].astype(float)
            user_feature['46'] = user_feature['46'].astype(float)
            user_feature['ts_placed_percent'] = user_feature['ts_placed_percent'].astype(float)
           
 
            #user_feature = user_feature[['36', '46','date','ts_placed_percent']].fillna(0.0)
            user_feature = user_feature.set_index(['date'])
            user_feature = user_feature.sort_index()
            if user_feature.index[-1].strftime('%Y-%m-%d') == '2018-11-21':
                return {}
            else:
                return {uid : len(user_feature)}

    user_data = feature_rdd.groupBy(lambda x : x.uid).map(functools.partial(user_feature_analysis, trade_dates)).reduce(lambda u1, u2: {**u1, **u2})
    spark.stop()
    set_trace()

