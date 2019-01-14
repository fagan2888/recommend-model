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
def data(ctx):

    '''user analysis
    '''
    pass


@data.command()
@click.pass_context
def app_log_update(ctx):

    dates = pd.date_range('2017-01-01', datetime.now()).strftime('%Y-%m-%d').tolist()
    dates.sort()
    dirs = os.listdir('data/app_log')
    exited_dates = []
    for f in dirs:
        exited_dates.append(f[8:18])
    exited_dates.sort()
    exited_dates = exited_dates[0:-30]
    new_dates = list(set(dates).difference(set(exited_dates)))
    new_dates.sort()
    print(new_dates)

    def app_log_date_update(new_date):
        start_timestamp = time.mktime(time.strptime(new_date, '%Y-%m-%d')) * 1000
        end_timestamp = start_timestamp + 24 * 60 * 60 *1000
        query = {"query":{"range":{"c_time":{
                            "lt":int(end_timestamp),
                            "gt":int(start_timestamp)
                                    }
                            }
                        }
                }
        logs = ESData().load_access_data(query)
        f = open('data/app_log/app_log_%s' % new_date, 'w')
        for log in logs:
            f.write(str(log))
            f.write('\n')
        f.close()
        return None


    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[8]')
            .set("spark.executor.memory", "8G")
            .set('spark.driver.memory', '8G')
            .set('spark.driver.maxResultSize', '8G'))
    sc = SparkContext(conf=spark_conf)
    new_dates_rdd = sc.parallelize(new_dates)

    new_dates_rdd.map(app_log_date_update).collect()


    pass



@data.command()
@click.pass_context
def ts_order(ctx):

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
            .set("spark.executor.memory", "8G")
            .set('spark.driver.memory', '8G')
            .set('spark.driver.maxResultSize', '8G'))

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
    ts_order_rdd = df.rdd.repartition(128)

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
    ts_order_rdd = df.rdd.repartition(128)

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
    ts_order_df.to_csv('data/ts_order.csv')


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

    ts_holding_nav.to_csv('data/ts_holding_nav.csv')


@data.command()
@click.pass_context
def user_question_answer(ctx):

    #engine = database.connection('recommend')
    #Session = sessionmaker(bind=engine)
    #session = Session()
    #sql = session.query(recommend.user_questionnaire_answers.uq_uid, recommend.user_questionnaire_answers.uq_questionnaire_id, recommend.user_questionnaire_answers.uq_question_id, recommend.user_questionnaire_answers.uq_answer, recommend.user_questionnaire_answers.uq_question_type, recommend.user_questionnaire_answers.uq_start_time, recommend.user_questionnaire_answers.uq_end_time).statement
    #user_question_answer_df = pd.read_sql(sql, session.bind, index_col = ['uq_uid','uq_questionnaire_id'], parse_dates = ['uq_start_time', 'uq_end_time'])
    #session.commit()
    #session.close()
    #user_question_answer_df.to_csv('tmp/user_question_answer.csv')

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
            if not k.startswith('100'):
                return pd.DataFrame()
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


    user_question_answer_df = pd.concat(user_question_answer_rdd.groupBy(lambda row : row.uq_uid).map(question_answer_feature).collect())
    print(user_question_answer_df.tail())
    user_question_answer_df.to_csv('tmp/user_question_answer_feature.csv')

    set_trace()


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
    df.to_csv('tmp/wechat_users.csv')


@data.command()
@click.pass_context
def yingmi_user_account(ctx):

    engine = database.connection('trade')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(trade.yingmi_accounts.ya_uid, trade.yingmi_accounts.ya_name).statement
    df = pd.read_sql(sql, session.bind, index_col = ['ya_uid'])
    session.commit()
    session.close()
    df.to_csv('tmp/user_name.csv')


@data.command()
@click.pass_context
def user_log(ctx):

    ts_order = pd.read_csv('./data/ts_order.csv', index_col = 'ts_order_uid')
    ts_order_uids = set(ts_order.index)

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
                    .set("spark.executor.memory", "32G")
                            .set('spark.driver.memory', '32G')
                                    .set('spark.driver.maxResultSize', '32G'))

    sc = SparkContext(conf=spark_conf)
    log_rdd = sc.textFile('data/app_log/app_log_*')
    log_rdd = log_rdd.repartition(480)


    def log_item(valid_uids, x):
        try:
            x = eval(x)
            x = x['_source']
            #用户不在购买用户中，直接过滤掉
            if(int(x['uid']) not in valid_uids):
                return (-1, {})
            #某只基金详情页
            elif x['page_id'] == '2032':
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d") :{'page_id=' + x['page_id']: 1}})
            #调整风险等级页
            elif x['ref_id'] == '2147':
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d") :{'ref_id=' + x['ref_id']: 1}})
            #我的资产页面
            elif x['page_id'] == '2013':
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d") :{'page_id=' + x['page_id']: 1}})
            #收益详情页
            elif x['page_id'] == '2162':
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d") :{'page_id=' + x['page_id']: 1}})
            #首页
            elif x['page_id'] == '3078':
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d") :{'page_id=' + x['page_id']: 1}})
            #资产配置详情页
            elif x['page_id'] == '2006':
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d") :{'page_id=' + x['page_id']: 1}})
            #点击追加购买按钮
            elif (x['page_id'] == '2013') and (x['ctrl'] == '4'):
                return (x['uid'], { datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d") :{
                      'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl']: 1}})
            #点击赎回按钮
            elif (x['page_id'] == '2013') and (x['ctrl'] == '6'):
                return (x['uid'], { datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl']: 1}})
            #点击持有基金按钮
            elif (x['page_id'] == '2148') and (x['ctrl'] == '1') and (x['ref_id'] == '2013') and (x['oid'] == '20'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d") :{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
            #点击投资分析按钮
            elif (x['page_id'] == '2126') and (x['ctrl'] == '8') and (x['ref_id'] == '2013') and (x['oid'] == '20'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d") :{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
            #点击交易记录按钮
            elif (x['page_id'] == '2150') and (x['ctrl'] == '2') and (x['ref_id'] == '2013') and (x['oid'] == '20'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d") :{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
            #点击调仓信息按钮
            elif (x['page_id'] == '2127') and (x['ctrl'] == '3') and (x['ref_id'] == '2013') and (x['oid'] == '20'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
            #点击定投管理按钮
            elif (x['page_id'] == '2133') and (x['ctrl'] == '4') and (x['ref_id'] == '2013') and (x['oid'] == '20'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d") :{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
            #邀请好友按钮
            elif (x['page_id'] == '2154') and (x['ctrl'] == '5') and (x['ref_id'] == '2013') and (x['oid'] == '20'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
            #微信通知按钮
            elif (x['page_id'] == '2171') and (x['ctrl'] == '9') and (x['ref_id'] == '2013') and (x['oid'] == '20'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
            #联系客服
            elif (x['page_id'] == '2161'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                       'page_id=' + x['page_id']: 1}})
            #智能组合页面-历史业绩
            elif (x['page_id'] == '2139') and (x['ctrl'] == '2') and (x['ref_id'] == '2006') and (x['oid'] == '1'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
            #智能组合页面-未来预期
            elif (x['page_id'] == '2146') and (x['ctrl'] == '2') and (x['ref_id'] == '2006') and (x['oid'] == '2'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
            #智能组合页面-风险控制
            elif (x['page_id'] == '2141') and (x['ctrl'] == '2') and (x['ref_id'] == '2006') and (x['oid'] == '3'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
            #智能组合页面-全球配置
            elif (x['page_id'] == '2142') and (x['ctrl'] == '2') and (x['ref_id'] == '2006') and (x['oid'] == '4'):
                #print(x['uid'], {datetime.fromtimestamp(int(x['@timestamp']) / 1000).strftime("%Y-%m-%d"):{
                #      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&oid=' + x['oid']: 1}})
            #智能组合页面-动态调仓
            elif (x['page_id'] == '2143') and (x['ctrl'] == '2') and (x['ref_id'] == '2006') and (x['oid'] == '5'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl'] + '&' + 'oid=' + x['oid']: 1}})
            #智能组合页面-立即购买
            elif (x['page_id'] == '2006') and (x['ctrl'] == '4'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl']: 1}})
            #智能组合页面-查看配置详情
            elif (x['page_id'] == '2138') and (x['ctrl'] == '3') and (x['ref_id'] == '2006'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl']: 1}})
            #查看配置详情-立即购买
            elif (x['page_id'] == '2110') and (x['ctrl'] == '2') and (x['ref_id'] == '2138'):
                return (x['uid'], {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{
                      'ref_id=' + x['ref_id'] + '&' + 'page_id=' + x['page_id'] + '&' + 'ctrl=' + x['ctrl']: 1}})
            return (-1, {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{'no_sense_action' : 1}})
            #return NULL
        except:
            return (-1, {datetime.fromtimestamp(int(x['c_time']) / 1000).strftime("%Y-%m-%d"):{'except' : 1}})


    def log_user(log1, log2):
        for date in log1.keys():
            date_log2 = log2.setdefault(date, {})
            date_log1 = log1[date]
            for ctrl in date_log1.keys():
                date_log2_ctrl = date_log2.setdefault(ctrl, 0)
                date_log2[ctrl] = date_log2_ctrl + date_log1[ctrl]  
        return log2


    def log_user_feature(x):
        uid = x[0]
        data = x[1]
        if (uid == '') or (uid == -1):
            return pd.DataFrame()
        log_df = pd.DataFrame(data).T.fillna(0.0)
        log_df.index.name = 'date'
        log_df = log_df.reset_index()
        log_df['uid'] = uid
        log_df = log_df.set_index(['uid', 'date'])
        log_df = log_df.sort_index()
        if len(log_df) == 1:
            return pd.DataFrame()
        return log_df


    logs = log_rdd.map(functools.partial(log_item, ts_order_uids)).reduceByKey(log_user) \
                    .map(log_user_feature).reduce(lambda a, b : pd.concat([a,b], axis = 0, sort = True)).fillna(0.0)
    sc.stop()
    print(logs.tail())
    logs.to_csv('data/log_feature.csv')



@data.command()
@click.pass_context
def all_feature(ctx):

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
            .set("spark.executor.memory", "48G")
            .set('spark.driver.memory', '48G')
            .set('spark.driver.maxResultSize', '48G'))

    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    ts_order_df = spark.read.csv("data/ts_order.csv", header = True)
    ts_holding_df = spark.read.csv("data/ts_holding_nav.csv", header = True)
    log_df = spark.read.csv('data/log_feature.csv', header = True)


    feature_rdd = ts_holding_df.rdd
    feature_rdd = feature_rdd.union(log_df.rdd)
    feature_rdd = feature_rdd.union(ts_order_df.rdd)

    feature_rdd = feature_rdd.repartition(1000)

    def combine_rdd(x):
        v = x.asDict()
        if 'uid'in v:
            v['feature_uid'] = v['uid']
            v['feature_date'] = v['date']
        elif 'ts_holding_uid' in v:
            v['feature_uid'] = v['ts_holding_uid']
            v['feature_date'] = v['ts_date']
        elif 'ts_order_uid' in v:
            v['feature_uid'] = v['ts_order_uid']
            v['feature_date'] = v['ts_trade_date']

        return v


    #def xirr(transactions):
    #    years = [(ta[0] - transactions[0][0]) / 365.0 for ta in transactions]
    #    residual = 1
    #    step = 0.05
    #    guess = 0.05
    #    epsilon = 0.0001
    #    limit = 10000
    #    while abs(residual) > epsilon and limit > 0:
    #        limit -= 1
    #        residual = 0.0
    #        for i, ta in enumerate(transactions):
    #            residual += ta[1] / pow(guess, years[i])
    #        if abs(residual) > epsilon:
    #            if residual > 0:
    #                guess += step
    #            else:
    #                guess -= step
    #                step /= 2.0
    #    #print(transactions, guess - 1)
    #    return guess-1

    def user_feature(uids, x):

        uid = x[0]

        if int(uid) not in uids:
            return pd.DataFrame()

        vs = x[1]


        data = {}
        for item in list(vs):
            item_date = item['feature_date']
            item_data = data.setdefault(item_date, {})
            for k in item.keys():
                item_data[k] = item[k]

        v = pd.DataFrame(data).T
        v = v.sort_index()

        ts_dates = v.ts_date.dropna().tolist() + v.ts_trade_date.dropna().tolist()


        v.index = pd.to_datetime(v.index.ravel())
        v = v.reindex(pd.date_range(min(ts_dates), max(ts_dates)))

        v.ts_risk = v.ts_risk.astype(np.float32)
        v.ts_profit = v.ts_profit.astype(np.float32)
        v.ts_asset = v.ts_asset.astype(np.float32)
        v.ts_nav = v.ts_nav.astype(np.float32)
        v.ts_placed_percent = v.ts_placed_percent.astype(np.float32)
        v.ts_placed_amount = v.ts_placed_amount.astype(np.float32)
        v.ts_processing_asset = v.ts_processing_asset.astype(np.float32)
        v['ts_total_asset'] = v.ts_asset + v.ts_processing_asset

        v.ts_risk = v.ts_risk.fillna(method = 'pad')
        v.ts_profit = v.ts_profit.fillna(method = 'pad')
        v.ts_asset = v.ts_asset.fillna(method = 'pad')
        v.ts_nav = v.ts_nav.fillna(method = 'pad')

        v.index.name = 'date'
        v.ts_trade_type_status = v.ts_trade_type_status.astype(np.float32)
        if len(v.ts_nav.dropna()) == 0:
            return pd.DataFrame()
        #有的数据第一条记录为nan
        v.ts_nav.iloc[0] = 1.0
        if len(v.ts_trade_type_status.dropna()) <= 0:
            return pd.DataFrame()
        #把最后一次赎回以后的净值数据去掉
        if v.ts_trade_type_status.dropna().iloc[-1] == 46.0:
            #print(v.ts_trade_type_status.dropna().index[-1])
            max_date = v.ts_trade_type_status.dropna().index[-1]
            v = v[v.index <= max_date]
        #v = v[['36', '46', '56', '66', 'ts_asset', 'ts_nav', 'ts_placed_amount', 'ts_placed_percent', 'ts_processing_asset', 'ts_profit', 'ts_risk']]

        try:
            #有的用户没有Log日志,就没有'uid'和'date'类,分类时过滤掉这些用户
            v = v.drop(['feature_date', 'ts_order_uid', 'ts_holding_uid', 'feature_uid', 'uid', 'date', 'ts_trade_date', 'ts_date'], axis = 1)
            #v = v.drop(['feature_date', 'ts_order_uid', 'ts_holding_uid', 'feature_uid', 'ts_trade_date', 'ts_date'], axis = 1)
        except:
            return pd.DataFrame()

        v['uid'] = uid
        v = v.reset_index()
        v = v.set_index(['uid', 'date'])
        v = v.sort_index()
        #v = v.iloc[0:-1]

        #print(v.columns)
        #trans = v[['ts_trade_type_status', 'ts_placed_amount', 'ts_total_asset', 'ts_placed_percent']]
        #trans.index = trans.index.get_level_values(1)
        #xirrs = []
        #for d in trans.index:
        #    current_trans = trans[trans.index <= d]
        #    data = []
        #    for i in range(0, len(current_trans)):
        #        date = current_trans.index[i]
        #        if current_trans.loc[date, 'ts_trade_type_status'] == 36.0:
        #            data.append([i, -1.0 * current_trans.loc[date, 'ts_placed_amount']])
        #        elif current_trans.loc[date, 'ts_trade_type_status'] == 56.0:
        #            data.append([i, -1.0 * current_trans.loc[date, 'ts_placed_amount']])
        #        elif current_trans.loc[date, 'ts_trade_type_status'] == 46.0:
        #            data.append([i, current_trans.loc[date, 'ts_placed_percent'] * current_trans.iloc[i - 1].ts_total_asset])
        #    data.append([len(current_trans) - 1 , current_trans.iloc[-1].ts_total_asset])
        #    #if data[0][0] > 0:
        #    #    print(uid, current_trans)
        #    _xirr = xirr(data)
        #    #if _xirr == -1.0:
        #    #    print(uid, data)
        #    xirrs.append(_xirr)
        #    #print(xirrs)
        #    #current_trans_df = pd.DataFrame(np.array(data).T)
        #    #print(current_trans_df)
        #v['xirrs'] = xirrs
        #print(v[v.ts_trade_type_status.isin([46.0, 36.0, 56.0])][['ts_trade_type_status', 'ts_placed_amount']])
        #print(v.tail())
        #过滤掉用户购买之前的log日志
        #v.ts_nav = v.ts_nav.fillna(method = 'pad')
        #v = v[pd.isnull(v.ts_nav)]
        #if len(v) >= 1:
        #    print(v)
        #print(v.ts_nav.notnull())


        return v


    #features = feature_rdd.map(combine_rdd).groupBy(lambda x : x['ts_holding_uid']).map(user_feature).collect()
    ts_order_df = pd.read_csv('tmp/ts_order.csv', index_col = ['ts_order_uid'])
    ts_holding_df = pd.read_csv('tmp/ts_holding_nav.csv', index_col = ['ts_holding_uid'])
    ts_order_uids = set(ts_order_df.index)
    ts_holding_uids = set(ts_holding_df.index)
    order_holding_uids = ts_order_uids & ts_holding_uids
    #print(ts_order_df.tail())

    features = feature_rdd.map(combine_rdd).groupBy(lambda x : x['feature_uid']).map(functools.partial(user_feature, order_holding_uids)).collect()
    feature_df = pd.concat(features, axis = 0)

    print(feature_df.tail())
    feature_df.to_csv('data/all_feature.csv')



@data.command()
@click.pass_context
def user_redeem_train_feature(ctx):


    index_sh = base_ra_index_nav.load_series('120000016')
    index_sh.index.name = 'td_date'
    trade_dates = base_trade_dates.load_index()

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
            .set("spark.executor.memory", "24G")
            .set('spark.driver.memory', '24G')
            .set('spark.driver.maxResultSize', '24G'))


    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    feature = spark.read.csv("data/all_feature.csv", header = True)
    feature_rdd = feature.rdd.repartition(1000)


    #构建用户特征用于分析
    def user_feature_analysis(trade_dates, index_sh, x):

        
        uid = x[0]
        
        user_feature = pd.DataFrame([v.asDict() for v in x[1]]).fillna(np.nan)
        user_feature['date'] = pd.to_datetime(user_feature.date)
        user_feature = user_feature[user_feature.date.isin(trade_dates)]
        user_feature = user_feature.set_index(['uid', 'date'])
        user_feature = user_feature.sort_index()
        index_sh = index_sh.sort_index()
        
        
        info_cols = ['ts_nav', 'ts_asset', 'ts_profit', 'ts_risk', 'ts_total_asset']
        for col in info_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(method = 'pad')
        buy_redeem_cols = [ '36', '46', '56', '66']
        for col in buy_redeem_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(0.0)
        
        log_cols = ['page_id=2006', 'page_id=2013', 'page_id=2032', 'page_id=2161', 'page_id=3078', 'ref_id=2006&page_id=2138&ctrl=3',\
            'ref_id=2006&page_id=2139&ctrl=2&oid=1', 'ref_id=2006&page_id=2141&ctrl=2&oid=3', 'ref_id=2006&page_id=2142&oid=4', \
            'ref_id=2006&page_id=2143&ctrl=2&oid=5', 'ref_id=2006&page_id=2146&ctrl=2&oid=2',\
            'ref_id=2013&page_id=2126&ctrl=8&oid=20', 'ref_id=2013&page_id=2127&ctrl=3&oid=20', \
            'ref_id=2013&page_id=2133&ctrl=4&oid=20', 'ref_id=2013&page_id=2148&ctrl=1&oid=20', \
            'ref_id=2013&page_id=2150&ctrl=2&oid=20', 'ref_id=2013&page_id=2154&ctrl=5&oid=20',\
            'ref_id=2013&page_id=2171&ctrl=9&oid=20', 'ref_id=2138&page_id=2110&ctrl=2', 'ref_id=2147']
        
        for col in log_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(0.0)
        
        tag = np.zeros(len(user_feature))
        
        #二类分类研究购买 
        tag[user_feature['36'] == 1.0] = 0.0
        tag[user_feature['46'] == 1.0] = 1.0
        
        user_feature['tag'] = tag
        user_feature['tag'] = user_feature.tag.shift(-1)
        user_feature = user_feature.iloc[0:-1]
        
        
        #计算当前回撤和最大回撤
        user_feature['drawdown'] = 1.0 - user_feature.ts_nav / user_feature.ts_nav.cummax()
        user_feature['max_drawdown'] = user_feature.drawdown.cummax()
        user_feature['profit_drawdown'] = 1.0 - user_feature.ts_profit / user_feature.ts_profit.cummax()
        user_feature['total_asset_drawdown'] = 1.0 - user_feature.ts_total_asset / user_feature.ts_total_asset.cummax()
        
        
        user_feature = user_feature[['ts_nav', 'ts_total_asset', 'tag', '36', '46', '56', '66', 'ts_profit', \
                        'ts_risk', 'drawdown', 'max_drawdown', 'profit_drawdown',\
                        'total_asset_drawdown'] + log_cols].fillna(0.0)
        
        
        user_feature['inc'] = user_feature.ts_nav.pct_change().fillna(0.0)
        
        index_sh = index_sh.loc[user_feature.index.get_level_values(1)]
        index_sh_inc = index_sh.pct_change().fillna(0.0)
        user_feature['inc_minus_sh'] = user_feature.inc.ravel() - index_sh_inc.ravel()
        user_feature['index_sh_inc'] = index_sh_inc.ravel()
        
        
        user_feature['days'] = np.arange(1, len(user_feature) + 1)
        
        
        days = [5, 10, 20, 60, 120, 200, 300]
        
        for day in days:
            
            user_feature['inc_%dd' % day] = user_feature.ts_nav.pct_change(day).fillna(0.0).astype(np.float32)
            user_feature['std_%dd' % day] = user_feature.ts_nav.pct_change().rolling(day).std().fillna(0.0).astype(np.float32)
            user_feature['shape_%dd' % day] = (user_feature['inc_%dd' % day] / user_feature['std_%dd' % day]).fillna(0.0).astype(np.float32)
            
            user_feature['sh_inc_%dd' % day] = user_feature.index_sh_inc.rolling(day).mean().fillna(0.0).astype(np.float32)
            user_feature['sh_std_%dd' % day] = user_feature.index_sh_inc.rolling(day).std().fillna(0.0).astype(np.float32)
            user_feature['sh_shape_%dd' % day] = (user_feature['sh_inc_%dd' % day] / user_feature['sh_std_%dd' % day]).fillna(0.0).astype(np.float32)
            
            user_feature['inc_minus_sh_%dd' % day] = user_feature.inc_minus_sh.rolling(day).mean().fillna(0.0).astype(np.float32)
            user_feature['inc_minus_sh_std_%dd' % day] = user_feature.inc_minus_sh.rolling(day).std().fillna(0.0).astype(np.float32)
            
            user_feature['36_%dd' % day] = user_feature['36'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['46_%dd' % day] = user_feature['46'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['56_%dd' % day] = user_feature['56'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['66_%dd' % day] = user_feature['66'].rolling(day).sum().fillna(0.0).astype(np.float32)
            
            user_feature['ts_profit_%dd' % day] = user_feature['ts_profit'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['ts_total_%dd' % day] = user_feature.ts_total_asset.rolling(day).mean().fillna(0.0).astype(np.float32)

        
            for log_col in log_cols:
                user_feature[log_col + '_%dd' % day] = user_feature[log_col].rolling(day).mean().fillna(0.0).astype(np.float32)
        
        #unixtimes = []
        #for d in user_feature.index.get_level_values(1):
        #    unixtimes.append(time.mktime(d.timetuple()))
            
        #user_feature['unixtime'] = unixtimes
        
        cols = list(user_feature)
        cols.pop(cols.index('tag'))
        cols.append('tag')
        user_feature = user_feature[cols]
        
        user_feature = user_feature.fillna(0.0)
        user_feature[user_feature == np.inf] = 1000
        user_feature[user_feature == -np.inf] = -1000

        pd.Series(user_feature.columns).to_csv('data/redeem_feature_name.csv')
        numbers = np.random.randint(0, 20, len(user_feature))
        #print(tuple(zip(numbers, user_feature.values)))
        #print(user_feature)
        #return zip(numbers, user_feature.values)
        return tuple(zip(numbers, user_feature.values))

    
    def user_feature_reduce(v):
            num = v[0]
            data = list(v[1])
            hold = 0
            redeem = 0
            features = []
            for d in data:
                tag = d[1][-1]
                if tag == 0.0:
                    hold = hold + 1
                elif tag == 1.0:
                    redeem = redeem + 1
                features.append(d[1])

            print('seed', num, 'hold num', hold, 'redeem num', redeem)
            np.save('./feature_data/redeem_feature_' + str(data[0][0]), np.array(features))
            return


    data = feature_rdd.groupBy(lambda x : x.uid).flatMap(functools.partial(user_feature_analysis, trade_dates, index_sh)).groupBy(lambda x : x[0]).map(user_feature_reduce).collect()
    spark.stop()



@data.command()
@click.pass_context
def user_buy_train_feature(ctx):


    index_sh = base_ra_index_nav.load_series('120000016')
    index_sh.index.name = 'td_date'
    trade_dates = base_trade_dates.load_index()

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
            .set("spark.executor.memory", "24G")
            .set('spark.driver.memory', '24G')
            .set('spark.driver.maxResultSize', '24G'))


    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    feature = spark.read.csv("data/all_feature.csv", header = True)
    feature_rdd = feature.rdd.repartition(1000)


    #构建用户特征用于分析
    def user_feature_analysis(trade_dates, index_sh, x):

        
        uid = x[0]
        
        user_feature = pd.DataFrame([v.asDict() for v in x[1]]).fillna(np.nan)
        user_feature['date'] = pd.to_datetime(user_feature.date)
        user_feature = user_feature[user_feature.date.isin(trade_dates)]
        user_feature = user_feature.set_index(['uid', 'date'])
        user_feature = user_feature.sort_index()
        index_sh = index_sh.sort_index()
        
        
        info_cols = ['ts_nav', 'ts_asset', 'ts_profit', 'ts_risk', 'ts_total_asset']
        for col in info_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(method = 'pad')
        buy_redeem_cols = [ '36', '46', '56', '66']
        for col in buy_redeem_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(0.0)
        
        log_cols = ['page_id=2006', 'page_id=2013', 'page_id=2032', 'page_id=2161', 'page_id=3078', 'ref_id=2006&page_id=2138&ctrl=3',\
            'ref_id=2006&page_id=2139&ctrl=2&oid=1', 'ref_id=2006&page_id=2141&ctrl=2&oid=3', 'ref_id=2006&page_id=2142&oid=4', \
            'ref_id=2006&page_id=2143&ctrl=2&oid=5', 'ref_id=2006&page_id=2146&ctrl=2&oid=2',\
            'ref_id=2013&page_id=2126&ctrl=8&oid=20', 'ref_id=2013&page_id=2127&ctrl=3&oid=20', \
            'ref_id=2013&page_id=2133&ctrl=4&oid=20', 'ref_id=2013&page_id=2148&ctrl=1&oid=20', \
            'ref_id=2013&page_id=2150&ctrl=2&oid=20', 'ref_id=2013&page_id=2154&ctrl=5&oid=20',\
            'ref_id=2013&page_id=2171&ctrl=9&oid=20', 'ref_id=2138&page_id=2110&ctrl=2', 'ref_id=2147']
        
        for col in log_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(0.0)
        
        tag = np.zeros(len(user_feature))
        
        #二类分类研究购买 
        tag[user_feature['36'] == 1.0] = 1.0
        tag[user_feature['46'] == 1.0] = 0.0
        
        user_feature['tag'] = tag
        user_feature['tag'] = user_feature.tag.shift(-1)
        user_feature = user_feature.iloc[0:-1]
        
        
        #计算当前回撤和最大回撤
        user_feature['drawdown'] = 1.0 - user_feature.ts_nav / user_feature.ts_nav.cummax()
        user_feature['max_drawdown'] = user_feature.drawdown.cummax()
        user_feature['profit_drawdown'] = 1.0 - user_feature.ts_profit / user_feature.ts_profit.cummax()
        user_feature['total_asset_drawdown'] = 1.0 - user_feature.ts_total_asset / user_feature.ts_total_asset.cummax()
        
        
        user_feature = user_feature[['ts_nav', 'ts_total_asset', 'tag', '36', '46', '56', '66', 'ts_profit', \
                        'ts_risk', 'drawdown', 'max_drawdown', 'profit_drawdown',\
                        'total_asset_drawdown'] + log_cols].fillna(0.0)
        
        
        user_feature['inc'] = user_feature.ts_nav.pct_change().fillna(0.0)
        
        index_sh = index_sh.loc[user_feature.index.get_level_values(1)]
        index_sh_inc = index_sh.pct_change().fillna(0.0)
        user_feature['inc_minus_sh'] = user_feature.inc.ravel() - index_sh_inc.ravel()
        user_feature['index_sh_inc'] = index_sh_inc.ravel()
        
        
        user_feature['days'] = np.arange(1, len(user_feature) + 1)
        
        
        days = [5, 10, 20, 60, 120, 200, 300]
        
        for day in days:
            
            user_feature['inc_%dd' % day] = user_feature.ts_nav.pct_change(day).fillna(0.0).astype(np.float32)
            user_feature['std_%dd' % day] = user_feature.ts_nav.pct_change().rolling(day).std().fillna(0.0).astype(np.float32)
            user_feature['shape_%dd' % day] = (user_feature['inc_%dd' % day] / user_feature['std_%dd' % day]).fillna(0.0).astype(np.float32)
            
            user_feature['sh_inc_%dd' % day] = user_feature.index_sh_inc.rolling(day).mean().fillna(0.0).astype(np.float32)
            user_feature['sh_std_%dd' % day] = user_feature.index_sh_inc.rolling(day).std().fillna(0.0).astype(np.float32)
            user_feature['sh_shape_%dd' % day] = (user_feature['sh_inc_%dd' % day] / user_feature['sh_std_%dd' % day]).fillna(0.0).astype(np.float32)
            
            user_feature['inc_minus_sh_%dd' % day] = user_feature.inc_minus_sh.rolling(day).mean().fillna(0.0).astype(np.float32)
            user_feature['inc_minus_sh_std_%dd' % day] = user_feature.inc_minus_sh.rolling(day).std().fillna(0.0).astype(np.float32)
            
            user_feature['36_%dd' % day] = user_feature['36'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['46_%dd' % day] = user_feature['46'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['56_%dd' % day] = user_feature['56'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['66_%dd' % day] = user_feature['66'].rolling(day).sum().fillna(0.0).astype(np.float32)
            
            user_feature['ts_profit_%dd' % day] = user_feature['ts_profit'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['ts_total_%dd' % day] = user_feature.ts_total_asset.rolling(day).mean().fillna(0.0).astype(np.float32)

        
            for log_col in log_cols:
                user_feature[log_col + '_%dd' % day] = user_feature[log_col].rolling(day).mean().fillna(0.0).astype(np.float32)
        
        #unixtimes = []
        #for d in user_feature.index.get_level_values(1):
        #    unixtimes.append(time.mktime(d.timetuple()))
            
        #user_feature['unixtime'] = unixtimes
        
        cols = list(user_feature)
        cols.pop(cols.index('tag'))
        cols.append('tag')
        user_feature = user_feature[cols]
        
        user_feature = user_feature.fillna(0.0)
        user_feature[user_feature == np.inf] = 1000
        user_feature[user_feature == -np.inf] = -1000

        pd.Series(user_feature.columns).to_csv('data/buy_feature_name.csv')
        numbers = np.random.randint(0, 20, len(user_feature))
        #print(tuple(zip(numbers, user_feature.values)))
        #print(user_feature)
        #return zip(numbers, user_feature.values)
        return tuple(zip(numbers, user_feature.values))

    
    def user_feature_reduce(v):
            num = v[0]
            data = list(v[1])
            hold = 0
            buy = 0
            features = []
            for d in data:
                tag = d[1][-1]
                if tag == 0.0:
                    hold = hold + 1
                elif tag == 1.0:
                    buy = buy + 1
                features.append(d[1])

            print('seed', num, 'hold num', hold, 'buy num', buy)
            np.save('./feature_data/buy_feature_' + str(data[0][0]), np.array(features))
            return


    data = feature_rdd.groupBy(lambda x : x.uid).flatMap(functools.partial(user_feature_analysis, trade_dates, index_sh)).groupBy(lambda x : x[0]).map(user_feature_reduce).collect()
    spark.stop()



@data.command()
@click.pass_context
def user_buy_predict_feature(ctx):


    index_sh = base_ra_index_nav.load_series('120000016')
    index_sh.to_csv('index_sh.csv')
    index_sh.index.name = 'td_date'
    trade_dates = base_trade_dates.load_index()

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
            .set("spark.executor.memory", "24G")
            .set('spark.driver.memory', '24G')
            .set('spark.driver.maxResultSize', '24G'))


    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    feature = spark.read.csv("data/all_feature.csv", header = True)
    feature_rdd = feature.rdd.repartition(1000)


    #构建用户特征用于分析
    def user_feature_analysis(trade_dates, index_sh, x):
        
        uid = x[0]
        
        user_feature = pd.DataFrame([v.asDict() for v in x[1]]).fillna(np.nan)
        user_feature['date'] = pd.to_datetime(user_feature.date)
        user_feature = user_feature[user_feature.date.isin(trade_dates)]
        user_feature = user_feature.set_index(['uid', 'date'])
        user_feature = user_feature.sort_index()
        index_sh = index_sh.sort_index()
       
        
        info_cols = ['ts_nav', 'ts_asset', 'ts_profit', 'ts_risk', 'ts_total_asset']
        for col in info_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(method = 'pad')
        buy_redeem_cols = [ '36', '46', '56', '66']
        for col in buy_redeem_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(0.0)
        
        log_cols = ['page_id=2006', 'page_id=2013', 'page_id=2032', 'page_id=2161', 'page_id=3078', 'ref_id=2006&page_id=2138&ctrl=3',\
            'ref_id=2006&page_id=2139&ctrl=2&oid=1', 'ref_id=2006&page_id=2141&ctrl=2&oid=3', 'ref_id=2006&page_id=2142&oid=4', \
            'ref_id=2006&page_id=2143&ctrl=2&oid=5', 'ref_id=2006&page_id=2146&ctrl=2&oid=2',\
            'ref_id=2013&page_id=2126&ctrl=8&oid=20', 'ref_id=2013&page_id=2127&ctrl=3&oid=20', \
            'ref_id=2013&page_id=2133&ctrl=4&oid=20', 'ref_id=2013&page_id=2148&ctrl=1&oid=20', \
            'ref_id=2013&page_id=2150&ctrl=2&oid=20', 'ref_id=2013&page_id=2154&ctrl=5&oid=20',\
            'ref_id=2013&page_id=2171&ctrl=9&oid=20', 'ref_id=2138&page_id=2110&ctrl=2', 'ref_id=2147']
        
        for col in log_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(0.0)
        
        tag = np.zeros(len(user_feature))
        
        #二类分类研究购买 
        tag[user_feature['36'] == 1.0] = 1.0
        tag[user_feature['46'] == 1.0] = 0.0
        
        user_feature['tag'] = tag
        #user_feature['tag'] = user_feature.tag.shift(-1)
        #user_feature = user_feature.iloc[0:-1]
        
        
        #计算当前回撤和最大回撤
        user_feature['drawdown'] = 1.0 - user_feature.ts_nav / user_feature.ts_nav.cummax()
        user_feature['max_drawdown'] = user_feature.drawdown.cummax()
        user_feature['profit_drawdown'] = 1.0 - user_feature.ts_profit / user_feature.ts_profit.cummax()
        user_feature['total_asset_drawdown'] = 1.0 - user_feature.ts_total_asset / user_feature.ts_total_asset.cummax()
        
        
        user_feature = user_feature[['ts_nav', 'ts_total_asset', 'tag', '36', '46', '56', '66', 'ts_profit', \
                        'ts_risk', 'drawdown', 'max_drawdown', 'profit_drawdown',\
                        'total_asset_drawdown'] + log_cols].fillna(0.0)
        
        
        user_feature['inc'] = user_feature.ts_nav.pct_change().fillna(0.0)
        
        index_sh = index_sh.loc[user_feature.index.get_level_values(1)]
        index_sh_inc = index_sh.pct_change().fillna(0.0)
        user_feature['inc_minus_sh'] = user_feature.inc.ravel() - index_sh_inc.ravel()
        user_feature['index_sh_inc'] = index_sh_inc.ravel()
        
        
        user_feature['days'] = np.arange(1, len(user_feature) + 1)
        
        
        days = [5, 10, 20, 60, 120, 200, 300]
        
        for day in days:
            
            user_feature['inc_%dd' % day] = user_feature.ts_nav.pct_change(day).fillna(0.0).astype(np.float32)
            user_feature['std_%dd' % day] = user_feature.ts_nav.pct_change().rolling(day).std().fillna(0.0).astype(np.float32)
            user_feature['shape_%dd' % day] = (user_feature['inc_%dd' % day] / user_feature['std_%dd' % day]).fillna(0.0).astype(np.float32)
            
            user_feature['sh_inc_%dd' % day] = user_feature.index_sh_inc.rolling(day).mean().fillna(0.0).astype(np.float32)
            user_feature['sh_std_%dd' % day] = user_feature.index_sh_inc.rolling(day).std().fillna(0.0).astype(np.float32)
            user_feature['sh_shape_%dd' % day] = (user_feature['sh_inc_%dd' % day] / user_feature['sh_std_%dd' % day]).fillna(0.0).astype(np.float32)
            
            user_feature['inc_minus_sh_%dd' % day] = user_feature.inc_minus_sh.rolling(day).mean().fillna(0.0).astype(np.float32)
            user_feature['inc_minus_sh_std_%dd' % day] = user_feature.inc_minus_sh.rolling(day).std().fillna(0.0).astype(np.float32)
            
            user_feature['36_%dd' % day] = user_feature['36'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['46_%dd' % day] = user_feature['46'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['56_%dd' % day] = user_feature['56'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['66_%dd' % day] = user_feature['66'].rolling(day).sum().fillna(0.0).astype(np.float32)
            
            user_feature['ts_profit_%dd' % day] = user_feature['ts_profit'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['ts_total_%dd' % day] = user_feature.ts_total_asset.rolling(day).mean().fillna(0.0).astype(np.float32)

        
            for log_col in log_cols:
                user_feature[log_col + '_%dd' % day] = user_feature[log_col].rolling(day).mean().fillna(0.0).astype(np.float32)
        
        #unixtimes = []
        #for d in user_feature.index.get_level_values(1):
        #    unixtimes.append(time.mktime(d.timetuple()))
            
        #user_feature['unixtime'] = unixtimes
        
        cols = list(user_feature)
        cols.pop(cols.index('tag'))
        cols.append('tag')
        user_feature = user_feature[cols]
        
        user_feature = user_feature.fillna(0.0)
        user_feature[user_feature == np.inf] = 1000
        user_feature[user_feature == -np.inf] = -1000

        yesterday = datetime.today() - timedelta(days=1) 
        if yesterday.date() in user_feature.index.get_level_values(1):
            user_feature = user_feature[user_feature.index.get_level_values(1) == yesterday.strftime('%Y-%m-%d')]
            return user_feature
        else:
            return pd.DataFrame()
        #print(user_feature)
        #pd.Series(user_feature.columns).to_csv('data/buy_feature_name.csv')
        #numbers = np.random.randint(0, 20, len(user_feature))
        #print(tuple(zip(numbers, user_feature.values)))
        #print(user_feature)
        #return zip(numbers, user_feature.values)
        return user_feature


    data = feature_rdd.groupBy(lambda x : x.uid).map(functools.partial(user_feature_analysis, trade_dates, index_sh)).reduce(lambda a, b : pd.concat([a,b], axis = 0))
    spark.stop()

    data.index.names = ['uid','date']
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    data.to_csv('predict_feature_data/buy_predict_feature_' + yesterday + '.csv')

@data.command()
@click.pass_context
def user_redeem_predict_feature(ctx):


    index_sh = base_ra_index_nav.load_series('120000016')
    index_sh.index.name = 'td_date'
    trade_dates = base_trade_dates.load_index()

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[24]')
            .set("spark.executor.memory", "24G")
            .set('spark.driver.memory', '24G')
            .set('spark.driver.maxResultSize', '24G'))


    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    feature = spark.read.csv("data/all_feature.csv", header = True)
    feature_rdd = feature.rdd.repartition(1000)


    #构建用户特征用于分析
    def user_feature_analysis(trade_dates, index_sh, x):
        
        uid = x[0]
        
        user_feature = pd.DataFrame([v.asDict() for v in x[1]]).fillna(np.nan)
        user_feature['date'] = pd.to_datetime(user_feature.date)
        user_feature = user_feature[user_feature.date.isin(trade_dates)]
        user_feature = user_feature.set_index(['uid', 'date'])
        user_feature = user_feature.sort_index()
        index_sh = index_sh.sort_index()
       
        
        info_cols = ['ts_nav', 'ts_asset', 'ts_profit', 'ts_risk', 'ts_total_asset']
        for col in info_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(method = 'pad')
        buy_redeem_cols = [ '36', '46', '56', '66']
        for col in buy_redeem_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(0.0)
        
        log_cols = ['page_id=2006', 'page_id=2013', 'page_id=2032', 'page_id=2161', 'page_id=3078', 'ref_id=2006&page_id=2138&ctrl=3',\
            'ref_id=2006&page_id=2139&ctrl=2&oid=1', 'ref_id=2006&page_id=2141&ctrl=2&oid=3', 'ref_id=2006&page_id=2142&oid=4', \
            'ref_id=2006&page_id=2143&ctrl=2&oid=5', 'ref_id=2006&page_id=2146&ctrl=2&oid=2',\
            'ref_id=2013&page_id=2126&ctrl=8&oid=20', 'ref_id=2013&page_id=2127&ctrl=3&oid=20', \
            'ref_id=2013&page_id=2133&ctrl=4&oid=20', 'ref_id=2013&page_id=2148&ctrl=1&oid=20', \
            'ref_id=2013&page_id=2150&ctrl=2&oid=20', 'ref_id=2013&page_id=2154&ctrl=5&oid=20',\
            'ref_id=2013&page_id=2171&ctrl=9&oid=20', 'ref_id=2138&page_id=2110&ctrl=2', 'ref_id=2147']
        
        for col in log_cols:
            user_feature[col] = user_feature[col].astype(np.float32)
            user_feature[col] = user_feature[col].fillna(0.0)
        
        tag = np.zeros(len(user_feature))
        
        #二类分类研究购买 
        tag[user_feature['36'] == 1.0] = 0.0
        tag[user_feature['46'] == 1.0] = 1.0
        
        user_feature['tag'] = tag
        #user_feature['tag'] = user_feature.tag.shift(-1)
        #user_feature = user_feature.iloc[0:-1]
        
        
        #计算当前回撤和最大回撤
        user_feature['drawdown'] = 1.0 - user_feature.ts_nav / user_feature.ts_nav.cummax()
        user_feature['max_drawdown'] = user_feature.drawdown.cummax()
        user_feature['profit_drawdown'] = 1.0 - user_feature.ts_profit / user_feature.ts_profit.cummax()
        user_feature['total_asset_drawdown'] = 1.0 - user_feature.ts_total_asset / user_feature.ts_total_asset.cummax()
        
        
        user_feature = user_feature[['ts_nav', 'ts_total_asset', 'tag', '36', '46', '56', '66', 'ts_profit', \
                        'ts_risk', 'drawdown', 'max_drawdown', 'profit_drawdown',\
                        'total_asset_drawdown'] + log_cols].fillna(0.0)
        
        
        user_feature['inc'] = user_feature.ts_nav.pct_change().fillna(0.0)
        
        index_sh = index_sh.loc[user_feature.index.get_level_values(1)]
        index_sh_inc = index_sh.pct_change().fillna(0.0)
        user_feature['inc_minus_sh'] = user_feature.inc.ravel() - index_sh_inc.ravel()
        user_feature['index_sh_inc'] = index_sh_inc.ravel()
        
        
        user_feature['days'] = np.arange(1, len(user_feature) + 1)
        
        
        days = [5, 10, 20, 60, 120, 200, 300]
        
        for day in days:
            
            user_feature['inc_%dd' % day] = user_feature.ts_nav.pct_change(day).fillna(0.0).astype(np.float32)
            user_feature['std_%dd' % day] = user_feature.ts_nav.pct_change().rolling(day).std().fillna(0.0).astype(np.float32)
            user_feature['shape_%dd' % day] = (user_feature['inc_%dd' % day] / user_feature['std_%dd' % day]).fillna(0.0).astype(np.float32)
            
            user_feature['sh_inc_%dd' % day] = user_feature.index_sh_inc.rolling(day).mean().fillna(0.0).astype(np.float32)
            user_feature['sh_std_%dd' % day] = user_feature.index_sh_inc.rolling(day).std().fillna(0.0).astype(np.float32)
            user_feature['sh_shape_%dd' % day] = (user_feature['sh_inc_%dd' % day] / user_feature['sh_std_%dd' % day]).fillna(0.0).astype(np.float32)
            
            user_feature['inc_minus_sh_%dd' % day] = user_feature.inc_minus_sh.rolling(day).mean().fillna(0.0).astype(np.float32)
            user_feature['inc_minus_sh_std_%dd' % day] = user_feature.inc_minus_sh.rolling(day).std().fillna(0.0).astype(np.float32)
            
            user_feature['36_%dd' % day] = user_feature['36'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['46_%dd' % day] = user_feature['46'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['56_%dd' % day] = user_feature['56'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['66_%dd' % day] = user_feature['66'].rolling(day).sum().fillna(0.0).astype(np.float32)
            
            user_feature['ts_profit_%dd' % day] = user_feature['ts_profit'].rolling(day).sum().fillna(0.0).astype(np.float32)
            user_feature['ts_total_%dd' % day] = user_feature.ts_total_asset.rolling(day).mean().fillna(0.0).astype(np.float32)

        
            for log_col in log_cols:
                user_feature[log_col + '_%dd' % day] = user_feature[log_col].rolling(day).mean().fillna(0.0).astype(np.float32)
        
        #unixtimes = []
        #for d in user_feature.index.get_level_values(1):
        #    unixtimes.append(time.mktime(d.timetuple()))
            
        #user_feature['unixtime'] = unixtimes
        
        cols = list(user_feature)
        cols.pop(cols.index('tag'))
        cols.append('tag')
        user_feature = user_feature[cols]
        
        user_feature = user_feature.fillna(0.0)
        user_feature[user_feature == np.inf] = 1000
        user_feature[user_feature == -np.inf] = -1000

        yesterday = datetime.today() - timedelta(days=1) 
        if yesterday.date() in user_feature.index.get_level_values(1):
            user_feature = user_feature[user_feature.index.get_level_values(1) == yesterday.strftime('%Y-%m-%d')]
            return user_feature
        else:
            return pd.DataFrame()
        #user_feature = user_feature[user_feature.index.get_level_values(1) == '2018-11-08']
        #print(user_feature)
        #pd.Series(user_feature.columns).to_csv('data/buy_feature_name.csv')
        #numbers = np.random.randint(0, 20, len(user_feature))
        #print(tuple(zip(numbers, user_feature.values)))
        #print(user_feature)
        #return zip(numbers, user_feature.values)


    data = feature_rdd.groupBy(lambda x : x.uid).map(functools.partial(user_feature_analysis, trade_dates, index_sh)).reduce(lambda a, b : pd.concat([a,b], axis = 0))
    spark.stop()
    data.index.names = ['uid','date']
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    data.to_csv('predict_feature_data/redeem_predict_feature_' + yesterday + '.csv')
