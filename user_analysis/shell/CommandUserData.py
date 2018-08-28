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
from db import database, trade, recommend, tongji, mapi
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pickle
from tempfile import NamedTemporaryFile
from ipdb import set_trace
import warnings
warnings.filterwarnings("ignore")


import traceback, code

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def data(ctx):

    '''user analysis
    '''
    pass


@data.command()
@click.pass_context
def ts_order(ctx):

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[32]')
            .set("spark.executor.memory", "50G")
            .set('spark.driver.memory', '50G')
            .set('spark.driver.maxResultSize', '50G'))

    engine = database.connection('trade')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(trade.ts_order.ts_uid, trade.ts_order.ts_txn_id, trade.ts_order.ts_portfolio_id ,trade.ts_order.ts_trade_type, trade.ts_order.ts_trade_status, trade.ts_order.ts_trade_date, trade.ts_order.ts_risk, trade.ts_order.ts_placed_percent, trade.ts_order.ts_placed_amount).filter(trade.ts_order.ts_trade_type.in_([3,4,5,6])).statement
    ts_order_df = pd.read_sql(sql, session.bind)
    ts_order_df = ts_order_df.rename(columns = {'ts_uid':'ts_order_uid'})
    session.commit()
    session.close()

    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    df = spark.createDataFrame(ts_order_df)
    ts_order_rdd = df.rdd.repartition(1000)

    #按照订单groupby,提取用户一个订单的动作
    def order_action(x):
        k, vs = x[0], x[1]
        v = pd.DataFrame([v.asDict() for v in vs])
        v.ts_trade_type = v.ts_trade_type.astype(int)
        v.ts_trade_status = v.ts_trade_status.astype(int)
        v.ts_placed_amount = v.ts_placed_amount.astype(float)
        v.ts_placed_percent = v.ts_placed_percent.astype(float)
        #过滤掉老组合的交易记录
        v = v[[False if portfolio_id.startswith('ZH') else True for portfolio_id in v.ts_portfolio_id]]
        #过滤掉交易失败的交易记录
        v = v[v.ts_trade_status > 0]
        #过滤掉交易撤单的交易记录
        v = v[v.ts_trade_status != 9]
        #过滤掉调仓赎回老组合的交易记录
        v = v[v.ts_trade_type != 8]
        #确认中和部分成功算作确认成功，这里主要是用户有了动作，说明用户有意图
        v.ts_trade_status = v.ts_trade_status.replace(1, 6)
        v.ts_trade_status = v.ts_trade_status.replace(5, 6)
        if len(v) > 0:
            ts_placed_amount = v.ts_placed_amount.sum()
            v = v.iloc[0:1]
            v.ts_placed_amount = ts_placed_amount
            v['ts_trade_type_status'] = str(v.ts_trade_type.ravel()[0]) + str(v.ts_trade_status.ravel()[0])
            v = v.set_index(['ts_txn_id'])
            return v
        else:
            return pd.DataFrame()

    ts_order_df = ts_order_rdd.groupBy(lambda row : row.ts_txn_id).map(order_action).reduce(lambda a, b : pd.concat([a,b], axis = 0, sort = True))

    df = spark.createDataFrame(ts_order_df)
    ts_order_rdd = df.rdd.repartition(1000)

    def user_action(x):
        k, vs = x[0], x[1]
        v = pd.DataFrame([v.asDict() for v in vs])
        v = v.set_index('ts_trade_date')
        v.ts_placed_amount = v.ts_placed_amount.astype(float)
        v.ts_placed_percent = v.ts_placed_percent.astype(float)
        vs = []
        for k, v in v.groupby(level = [0]):
            v = v.reset_index()
            v = v.set_index(['ts_trade_type_status'])
            for k, v in v.groupby(level = [0]):
                #一次操作的总金额
                ts_placed_amount = v.ts_placed_amount.sum()
                #如果多次赎回，设为多次赎回的百分比
                ts_placed_percent = 1.0 - np.prod([1.0 - percent for percent in v.ts_placed_percent])
                v = v.iloc[0:1]
                v.ts_placed_amount = ts_placed_amount
                v.ts_placed_percent = ts_placed_percent
                v[k] = 1.0
                vs.append(v)
        if len(vs) == 0:
            return pd.DataFrame()
        df = pd.concat(vs, axis = 0, sort = True)
        df = df.reset_index()
        df = df.set_index(['ts_order_uid', 'ts_trade_date'])
        return df

    ts_order_df = ts_order_rdd.groupBy(lambda row : row.ts_order_uid).map(user_action).reduce(lambda a, b : pd.concat([a,b], axis = 0, sort = True)).fillna(0.0)


    print(ts_order_df.tail(20))
    ts_order_df.to_csv('tmp/ts_order.csv')


@data.command()
@click.pass_context
def ts_holding(ctx):

    engine = database.connection('trade')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(trade.ts_holding_nav.ts_uid, trade.ts_holding_nav.ts_portfolio_id, trade.ts_holding_nav.ts_date, trade.ts_holding_nav.ts_nav, trade.ts_holding_nav.ts_asset, trade.ts_holding_nav.ts_profit, trade.ts_holding_nav.ts_processing_asset).statement
    ts_holding_nav = pd.read_sql(sql, session.bind)
    ts_holding_nav = ts_holding_nav.rename(columns = {'ts_uid':'ts_holding_uid'})
    ts_holding_nav = ts_holding_nav.set_index(['ts_holding_uid'])
    session.commit()
    session.close()

    ts_holding_nav.to_csv('tmp/ts_holding_nav.csv')


@data.command()
@click.pass_context
def user_question_answer(ctx):

    engine = database.connection('recommend')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(recommend.user_questionnaire_answers.uq_uid, recommend.user_questionnaire_answers.uq_questionnaire_id, recommend.user_questionnaire_answers.uq_question_id, recommend.user_questionnaire_answers.uq_answer, recommend.user_questionnaire_answers.uq_question_type, recommend.user_questionnaire_answers.uq_start_time, recommend.user_questionnaire_answers.uq_end_time).statement
    user_question_answer_df = pd.read_sql(sql, session.bind, index_col = ['uq_uid','uq_questionnaire_id'], parse_dates = ['uq_start_time', 'uq_end_time'])
    session.commit()
    session.close()
    user_question_answer_df.to_csv('tmp/user_question_answer.csv')

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
            .set("spark.executor.memory", "50G")
            .set('spark.driver.memory', '50G')
            .set('spark.driver.maxResultSize', '50G')
            .set('spark.cores.max', 32))

    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    df = spark.read.csv("tmp/user_question_answer.csv", header = True)
    user_question_answer_rdd = df.rdd.repartition(1000)

    def question_answer_feature(v):
        try:
            k = v[0]
            vs = v[1]
            datas = [v.asDict() for v in vs]
            v = pd.DataFrame(datas).dropna()
            v.uq_questionnaire_id = v.uq_questionnaire_id.astype(int)
            v.uq_question_type = v.uq_question_type.astype(int)
            v = v[v.uq_questionnaire_id == max(v.uq_questionnaire_id)]
            #v = v[v.uq_question_type == 0]
            v.uq_question_id = v.uq_question_id.astype(str)
            v['uq_question_answer'] = v.uq_question_id + v.uq_answer
            v = v.drop(columns = ['uq_question_type', 'uq_start_time', 'uq_end_time' ,'uq_question_id', 'uq_answer', 'uq_questionnaire_id'])
            v = v.set_index(['uq_uid', 'uq_question_answer'])
            v['tag'] = 1.0
            v = v[~v.index.duplicated()]
            v = v.unstack().fillna(0.0)
            v.columns = v.columns.droplevel(0)
        except:
            return pd.DataFrame()
        return v


    user_question_answer_df = user_question_answer_rdd.groupBy(lambda row : row.uq_uid).map(question_answer_feature).reduce(lambda a, b : pd.concat([a,b], axis = 0, sort=True).fillna(0.0))
    print(user_question_answer_df.tail())
    user_question_answer_df.to_csv('tmp/user_question_answer_feature.csv')



@data.command()
@click.pass_context
def feature(ctx):

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[32]')
            .set("spark.executor.memory", "50G")
            .set('spark.driver.memory', '50G')
            .set('spark.driver.maxResultSize', '50G'))

    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    ts_order_df = spark.read.csv("tmp/ts_order.csv", header = True)
    ts_holding_df = spark.read.csv("tmp/ts_holding_nav.csv", header = True)

    feature = ts_holding_df.join(ts_order_df, [ts_holding_df.ts_holding_uid == ts_order_df.ts_order_uid, ts_holding_df.ts_date == ts_order_df.ts_trade_date], 'outer')
    feature_rdd = feature.rdd.repartition(1000)

    def combine_rdd(x):
        v = x.asDict()
        if v['ts_holding_uid'] is None:
            v['ts_holding_uid'] = v['ts_order_uid']
            v['ts_date'] = v['ts_trade_date']
        return v

    def user_feature(x):
        k = x[0]
        vs = x[1]
        v = pd.DataFrame(list(vs))
        v.ts_date = pd.to_datetime(v.ts_date)
        v = v.set_index(['ts_holding_uid', 'ts_date'])
        v = v.sort_index().fillna(np.nan)
        v.ts_risk = v.ts_risk.fillna(method = 'pad')
        v.ts_profit = v.ts_profit.fillna(method = 'pad')
        v.ts_asset = v.ts_asset.fillna(method = 'pad')
        v.ts_nav = v.ts_nav.fillna(method = 'pad')
        v.ts_trade_type_status = v.ts_trade_type_status.astype(float)
        if len(v.ts_nav.dropna()) == 0:
            return pd.DataFrame()
        #有的数据第一条记录为nan
        v.ts_nav.iloc[0] = 1.0
        if len(v.ts_trade_type_status.dropna()) <= 0:
            return pd.DataFrame()
        if v.ts_trade_type_status.dropna().iloc[-1] == 46.0:
            max_date = v.ts_trade_type_status.dropna().index[-1][1]
            v = v[v.index.get_level_values(1) <= max_date]
        nav = v.ts_nav
        #v = v[['36', '46', '56', '66', 'ts_asset', 'ts_nav', 'ts_placed_amount', 'ts_placed_percent', 'ts_processing_asset', 'ts_profit', 'ts_risk']]
        return v

    features = feature_rdd.map(combine_rdd).groupBy(lambda x : x['ts_holding_uid']).map(user_feature).collect()
    feature_df = pd.concat(features, axis = 0)
    print(feature_df.tail())
    feature_df.to_csv('tmp/feature.csv')
    #feature_df = feature_rdd.map(combine_rdd).groupBy(lambda x : x['ts_uid']).map(user_feature).reduce(lambda a, b : pd.concat([a,b], axis = 0, sort=True))
    #print(feature.ts_uid)
    #feature = feature.toPandas().fillna(np.nan).set_index(['ts_uid'])
    #print(feature.tail())
    #feature = feature.iloc[:,~feature.columns.duplicated()]
    #print(feature.loc['1000373998'].sort_index())
    #feature.to_csv('tmp/feature.csv')


@data.command()
@click.pass_context
def log_raw_apps(ctx):

    engine = database.connection('tongji')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tongji.log_raw_apps.lr_date, tongji.log_raw_apps.lr_time, tongji.log_raw_apps.lr_uid, tongji.log_raw_apps.lr_pid, tongji.log_raw_apps.lr_page,
            tongji.log_raw_apps.lr_ctrl, tongji.log_raw_apps.lr_oid, tongji.log_raw_apps.lr_tag, tongji.log_raw_apps.lr_ref, tongji.log_raw_apps.lr_ver,
            tongji.log_raw_apps.lr_chn, tongji.log_raw_apps.lr_os, tongji.log_raw_apps.lr_flag, tongji.log_raw_apps.lr_ev).statement
    log_raw_apps = pd.read_sql(sql, session.bind, index_col = ['lr_uid'], parse_dates = ['lr_date'])
    session.commit()
    session.close()

    log_raw_apps.tail()
    log_raw_apps.to_csv('tmp/log_raw_apps.csv')

    pass


@data.command()
@click.pass_context
def log_dirty_detection(ctx):

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[32]')
            .set("spark.executor.memory", "50G")
            .set('spark.driver.memory', '50G')
            .set('spark.driver.maxResultSize', '50G'))

    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    log_df = spark.read.csv("tmp/log_raw_apps.csv", header = True)


    log_rdd = log_df.rdd.repartition(1000)


    #def ver_page(iterator):
    #    ver_page_num = {}
    #    for x in iterator:
    #        #print(x.lr_ver, x.lr_page)
    #        pages = ver_page_num.setdefault(x.lr_ver, {})
    #        page_num = pages.setdefault(x.lr_page, 0)
    #        pages[x.lr_page] = page_num + 1
    #    return list(ver_page_num.items())


    #def ver_page_reduce(p1, p2):
    #    for page in p1:
    #        p2_num = p2.setdefault(page, 0)
    #        p2[page] = p2_num + p1[page]
    #    return p2


    #page = log_rdd.mapPartitions(ver_page).reduceByKey(ver_page_reduce).collect()
    #data = {}
    #for ver, page_num in page:
    #    page_num_ser = pd.Series(page_num)
    #    data[ver] = page_num_ser

    #page_num_df = pd.DataFrame(data).fillna(0.0)
    #page_num_df.columns = page_num_df.columns.astype(int)
    #ver_num_ser = page_num_df.sum(axis = 0)
    #ver_num_ser = ver_num_ser[ver_num_ser > 5000]
    #valid_ver = ver_num_ser.index
    ##7077版本是智能组合上线的版本，对应2016年9月以后
    ##7088版本是智能组合千人千面上线的版本，对应2017年5月以后
    #valid_ver = list(valid_ver[valid_ver >= 7088])
    #valid_ver.sort()
    #page_num_df = page_num_df[valid_ver]
    #page_ratio_df = page_num_df.div(page_num_df.sum(axis = 0))
    #print(page_ratio_df.tail())
    #page_ratio_df.to_csv('tmp/page_ratio.csv')


    def ver_ctrl(iterator):
        ver_ctrls_num = {}
        for x in iterator:
            ctrls = ver_ctrls_num.setdefault(x.lr_ver, {})
            ctrl_id = x.lr_ref + '_'+ x.lr_page + '_' + x.lr_ctrl + '_' +  x.lr_oid
            ctrl_num = ctrls.setdefault(ctrl_id, 0)
            ctrls[ctrl_id] = ctrl_num + 1
        return list(ver_ctrls_num.items())


    def ver_ctrl_reduce(c1, c2):
        for ctrl in c1:
            c2_num = c2.setdefault(ctrl, 0)
            c2[ctrl] = c2_num + c1[ctrl]
        return c2


    ctrl = log_rdd.mapPartitions(ver_ctrl).reduceByKey(ver_ctrl_reduce).collect()
    data = {}
    for ver, ctrl_num in ctrl:
        ctrl_num_ser = pd.Series(ctrl_num)
        data[ver] = ctrl_num_ser

    ctrl_num_df = pd.DataFrame(data).fillna(0.0)
    ctrl_num_df.columns = ctrl_num_df.columns.astype(int)
    ver_num_ser = ctrl_num_df.sum(axis = 0)
    ver_num_ser = ver_num_ser[ver_num_ser > 5000]
    valid_ver = ver_num_ser.index
    #7077版本是智能组合上线的版本，对应2016年9月以后
    #7088版本是智能组合千人千面上线的版本，对应2017年5月以后
    valid_ver = list(valid_ver[valid_ver >= 7088])
    valid_ver.sort()
    ctrl_num_df = ctrl_num_df[valid_ver]
    ctrl_ratio_df = ctrl_num_df.div(ctrl_num_df.sum(axis = 0))
    print(ctrl_ratio_df.tail())
    ctrl_ratio_df.to_csv('tmp/ctrl_ratio.csv')



@data.command()
@click.pass_context
def log_analysis(ctx):

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[32]')
            .set("spark.executor.memory", "50G")
            .set('spark.driver.memory', '50G')
            .set('spark.driver.maxResultSize', '50G'))

    sc = SparkContext(conf=spark_conf)
    log_rdd = sc.textFile('data/app_log.txt')
    log_rdd = log_rdd.repartition(1000)


    def ver_page(x):
        x = eval(x)
        try:
            x = x['_source']
            return (x['version'], {x['page_id']:1})
        except:
            return (-1, {-1:0})

    def ver_page_reduce(p1, p2):
        for page in p1:
            p2_num = p2.setdefault(page, 0)
            p2[page] = p2_num + p1[page]
        return p2

    pages = log_rdd.map(ver_page).reduceByKey(ver_page_reduce).collect()
    sc.stop()


    data = {}
    for ver, page_num in pages:
        page_num_ser = pd.Series(page_num)
        data[ver] = page_num_ser

    page_num_df = pd.DataFrame(data).fillna(0.0)
    #page_num_df.columns = page_num_df.columns.astype(int)
    ver_num_ser = page_num_df.sum(axis = 0)
    ver_num_ser = ver_num_ser[ver_num_ser > 5000]
    valid_ver = ver_num_ser.index
    print(valid_ver)
    #7077版本是智能组合上线的版本，对应2016年9月以后
    #7088版本是智能组合千人千面上线的版本，对应2017年5月以后
    #valid_ver = list(valid_ver[valid_ver >= 7088])
    valid_ver.sort()
    page_num_df = page_num_df[valid_ver]
    page_ratio_df = page_num_df.div(page_num_df.sum(axis = 0))
    print(page_ratio_df.tail())
    #page_ratio_df.to_csv('tmp/page_ratio.csv')


@data.command()
@click.pass_context
def wechat_user(ctx):

    engine = database.connection('mapi')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(mapi.wechat_users.uid, mapi.wechat_users.mobile, mapi.wechat_users.service_id).statement
    df = pd.read_sql(sql, session.bind, index_col = ['uid'])
    session.commit()
    session.close()
    print(df)
    df.to_csv('tmp/wechat_users.csv')
