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
import random

from datetime import datetime, timedelta
from dateutil.parser import parse
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from util import xdict
from util.xdebug import dd
from db import database, trade, recommend, tongji, mapi, base_ra_index_nav, base_trade_dates, passport, asset_allocation
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pickle
from tempfile import NamedTemporaryFile
import functools
from ipdb import set_trace
from esdata import ESData
import json
import warnings

from elasticsearch import helpers
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb


warnings.filterwarnings("ignore")


import traceback, code

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def gb(ctx):

    '''user analysis
    '''
    pass


@gb.command()
@click.pass_context
def user_feature_data(ctx):

    user_tag_df = user_tag()
    user_mobile_feature_df = user_mobile_feature(user_tag_df.index.ravel())
    h4uid_df = h4uid(user_tag_df.index.ravel())
    user_questionnaire_answers_df = user_questionnaire_answers(h4uid_df)

    #print(user_mobile_feature_df.tail())
    #user_tag_df = pd.read_csv('./data/mofang.user_tags.csv', index_col = ['uid'])
    #user_mobile_feature_df = pd.read_csv('./data/user_mobile.csv', index_col = ['uid'])
    #user_questionnaire_answers_df = pd.read_csv('./data/user_questionnaire_answers.csv', index_col = ['uid'])
    uids = user_tag_df.index & user_mobile_feature_df.index & user_questionnaire_answers_df.index

    user_tag_df = user_tag_df.loc[uids]
    user_mobile_feature_df = user_mobile_feature_df.loc[uids]
    user_questionnaire_answers_df = user_questionnaire_answers_df.loc[uids]

    feature_df = pd.concat([user_tag_df, user_mobile_feature_df, user_questionnaire_answers_df], axis = 1, join_axes = [user_tag_df.index])
    feature_df.index.name = 'uid'

    feature_df.to_csv('data/high_inferior_user_feature.csv')
    print(feature_df.tail(2))



def user_tag():

    es = Elasticsearch(['10.111.66.91'], http_auth=('elastic', 'eSl&aPs5t3i1c'), port='9200') 
    index = 'mofang.user_tags'
    doc = 'utags_doc'


    query = {"query":{"match_all":{}}}
    res = helpers.scan(es, query, '10m', index=index, doc_type=doc, raise_on_error=True, preserve_order=False, request_timeout=1200)
    data = {}
    for x in res:
        x = x['_source']
        if 'uid' in x:
            data[x['uid']] = pd.Series(x)
    df = pd.DataFrame(data).T
    tags = ['pc_risk', 'age', 'sex' ,'city', 'sg_amount', 'gm_risk']

    df = df[tags]
    df.index.name = 'uid'
    df = df.dropna(how='all')
    df = df.replace('', np.nan)
    df.index = df.index.astype(int)

    df.to_csv('data/mofang.user_tags.csv')
    return df


def h4uid(uids):

    engine = database.connection('recommend')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(recommend.user_risk_analyze_results.id, recommend.user_risk_analyze_results.ur_uid, recommend.user_risk_analyze_results.ur_nare_id).statement
    user_risk_analyze_results_df = pd.read_sql(sql, session.bind)
    user_risk_analyze_results_df.ur_uid = user_risk_analyze_results_df.ur_uid.astype(int)
    user_risk_analyze_results_df = user_risk_analyze_results_df.set_index(['ur_uid'])
    user_risk_analyze_results_df = user_risk_analyze_results_df.groupby(level = [0]).first()
    user_risk_analyze_results_df = user_risk_analyze_results_df[user_risk_analyze_results_df.index.get_level_values(0).isin(uids)]
    #print(user_risk_analyze_results_df.tail())
    #print(user_tag_df.index)
    user_risk_analyze_results_df = user_risk_analyze_results_df['ur_nare_id']
    user_risk_analyze_results_df = user_risk_analyze_results_df.reset_index()
    user_risk_analyze_results_df = user_risk_analyze_results_df.set_index(['ur_nare_id'])
    

    sql = session.query(recommend.user_questionnaire_summaries.id, recommend.user_questionnaire_summaries.uq_uid).statement
    user_questionnaire_summaries_df = pd.read_sql(sql, session.bind, index_col = ['id'])
    h4uid_df = pd.concat([user_risk_analyze_results_df, user_questionnaire_summaries_df],axis =1, join_axes = [user_risk_analyze_results_df.index]) 

    h4uid_df = h4uid_df.reset_index()
    h4uid_df = h4uid_df.set_index(['ur_uid'])
    h4uid_df = h4uid_df['uq_uid']

    session.commit()
    session.close()

    for uid in h4uid_df.index:
        v = h4uid_df.loc[uid]
        if np.isnan(v):
            h4uid_df.loc[uid] = uid

    h4uid_df.to_csv('data/h4uid.csv')

    return h4uid_df


def user_mobile_feature(uids):

    engine = database.connection('passport')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(passport.users).statement
    user_mobile_df = pd.read_sql(sql, session.bind, index_col = ['id'])

    session.commit()
    session.close()

    user_mobile_df = user_mobile_df.loc[uids]
    user_mobile_df = user_mobile_df.rename(columns = {'mobile_anonymous':'mobile_3', 'device_info':'phone'})

    phones = []
    for phone in user_mobile_df.phone:
        try:
            if isinstance(phone, str):
                ph = json.loads(phone)
                if 'manufacturer' in ph:
                    ph = json.loads(ph)
                    phones.append(ph['manufacturer'])
                else:
                    phones.append(np.nan)
            else:
                phones.append(np.nan)
        except Exception as e:
            phones.append(np.nan)
    user_mobile_df.phone = phones
    user_mobile_df.mobile_3 = user_mobile_df.mobile_3.apply(lambda x: x[0:3])

    user_mobile_df.index.name = 'uid'
    user_mobile_df.to_csv('data/user_mobile.csv')

    return user_mobile_df


def user_questionnaire_answers(h4uid_df):

    engine = database.connection('recommend')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(recommend.user_questionnaire_answers.uq_uid, recommend.user_questionnaire_answers.uq_question_id, \
        recommend.user_questionnaire_answers.uq_answer, recommend.user_questionnaire_answers.id).statement

    user_questionnaire_answers_df = pd.read_sql(sql, session.bind)
    user_questionnaire_answers_df = user_questionnaire_answers_df.sort_values(['id'])

    user_questionnaire_answers_df = user_questionnaire_answers_df[['uq_uid', 'uq_question_id','uq_answer']]
    user_questionnaire_answers_df = user_questionnaire_answers_df.set_index(['uq_uid', 'uq_question_id'])
    user_questionnaire_answers_df = user_questionnaire_answers_df.groupby(level = [0, 1]).last()

    user_questionnaire_answers_df = user_questionnaire_answers_df.fillna(np.nan).dropna()
    user_questionnaire_answers_df = user_questionnaire_answers_df.reset_index()
    user_questionnaire_answers_df.uq_answer = user_questionnaire_answers_df.uq_answer.apply(lambda x : str.upper(x))

    user_questionnaire_answers_df.uq_question_id = user_questionnaire_answers_df.uq_question_id.astype(str)
    user_questionnaire_answers_df['uq_question_id'] = 'question_' + user_questionnaire_answers_df['uq_question_id']
    user_questionnaire_answers_df = user_questionnaire_answers_df[['uq_uid', 'uq_question_id', 'uq_answer']]
    user_questionnaire_answers_df = user_questionnaire_answers_df.set_index(['uq_uid', 'uq_question_id'])
    user_questionnaire_answers_df = user_questionnaire_answers_df.unstack()
    user_questionnaire_answers_df.columns = user_questionnaire_answers_df.columns.droplevel(0)


    data = {}
    for uid in h4uid_df.index:
        _id = int(h4uid_df.loc[uid])
        if _id in user_questionnaire_answers_df.index:
            data[uid] = user_questionnaire_answers_df.loc[int(_id)]

    user_questionnaire_answers_df = pd.DataFrame(data).T
    user_questionnaire_answers_df.index.name = 'uid'
    user_questionnaire_answers_df.to_csv('data/user_questionnaire_answers.csv')

    session.commit()
    session.close()

    return user_questionnaire_answers_df




@gb.command()
@click.pass_context
def xgboost_purchased(ctx):

    high_quality_df = pd.read_csv('./data/yulai_high_quality_user.csv', encoding='gbk', index_col = ['uid'])
    inferior_df = pd.read_csv('./data/yulai_inferior_user.csv', encoding='gbk', index_col = ['uid'])
    high_quality_df = high_quality_df[inferior_df.columns]

    all_user_feature_df = pd.read_csv('./data/high_inferior_user_feature.csv', encoding='utf8', index_col = ['uid'])
    all_user_feature_df = all_user_feature_df.rename(columns = {'pc_risk':'answer_risk', 'sg_amount':'first_buy_amount', 'gm_risk':'buy_risk'})

    columns = high_quality_df.columns & all_user_feature_df.columns
    high_quality_df = high_quality_df[columns]
    inferior_df = inferior_df[columns]
    all_user_feature_df = all_user_feature_df[columns]
 
    #print(high_quality_df.columns)
    high_quality_df['tag'] = 1.0
    inferior_df['tag']= 0.0
    tag_feature_df = pd.concat([high_quality_df, inferior_df], axis = 0)
    tag = tag_feature_df['tag']
    tag_feature_df = tag_feature_df.drop(['tag'], axis = 1)
    tag_feature_df = pd.get_dummies(tag_feature_df)
    all_user_feature_df = pd.get_dummies(all_user_feature_df)


    cols = tag_feature_df.columns & all_user_feature_df.columns
    tag_feature_df = tag_feature_df[cols]
    all_user_feature_df = all_user_feature_df[cols]


    X_train, X_test, y_train, y_test = train_test_split(tag_feature_df, tag, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=23)


    param = {'max_depth': 3,
                 #'eta': 0.5,
                 'silent': 1,
                 'objective': 'binary:logistic',
                 'lambda':1,
                 #'alpha' : 0.1,
                 'subsample':0.5,
                 'colsample_bytree':0.5,
                 'scale_pos_weight': 1.0 * len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                }
    param['eval_metric'] = 'auc'


    from sklearn.metrics import classification_report
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    dtest = xgb.DMatrix(X_test)
    evallist = [(dtrain,'train'),(dval, 'val')]
    num_round = 10000
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds = 20)
    y_pred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    pred = y_pred.copy()
    threshold = 0.5
    pred[pred >= threshold] = 1.0
    pred[pred <= threshold] = 0.0
    print(classification_report(y_test, pred, digits = 6)) 


    all_user_feature_test = xgb.DMatrix(all_user_feature_df)
    y_pred = bst.predict(all_user_feature_test, ntree_limit=bst.best_ntree_limit) 

    all_user_tag_df = pd.DataFrame(y_pred, index = all_user_feature_df.index, columns = ['score'])
    all_user_tag_df.index.name = 'uid'
    print(all_user_tag_df.tail())
    print(all_user_tag_df.loc[1000000154]) #骄阳
    #print(all_user_tag_df.loc[1000105609])
    print(all_user_tag_df.loc[1000134808]) #家辉
    print(all_user_tag_df.loc[1000368551]) #新宇
    print(all_user_tag_df.loc[1000000087]) #高蓬
    print(all_user_tag_df.loc[1000000091]) #张亮
    print(all_user_tag_df.loc[1000179536]) #安东
    print(all_user_tag_df.loc[1000106528]) #马老师
    #print(all_user_tag_df.loc[1000117745]) #海勇

    all_user_tag_df.to_csv('./data/high_inferior_user_score.csv')


@gb.command()
@click.pass_context
def user_score_2_db(ctx):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(asset_allocation.high_quality_inferior_user.uid).statement
    uid_df = pd.read_sql(sql, session.bind, index_col = ['uid'])


    all_user_tag_df = pd.read_csv('./data/high_inferior_user_score.csv', index_col = ['uid'])

    all_user_tag_df = all_user_tag_df[~all_user_tag_df.index.isin(uid_df.index)]

    all_user_tag_df['service_score'] = all_user_tag_df['score']

    all_user_tag_df.service_score = all_user_tag_df.service_score.apply(lambda x: x if random.random() >= 0.3 else -1.0)

    records = []
    for uid in all_user_tag_df.index:
        high_quality_inferior_user = asset_allocation.high_quality_inferior_user()
        high_quality_inferior_user.uid = uid
        high_quality_inferior_user.score = all_user_tag_df.loc[uid, 'score']
        high_quality_inferior_user.service_score = all_user_tag_df.loc[uid, 'service_score']
        high_quality_inferior_user.created_at = datetime.now()
        high_quality_inferior_user.updated_at = datetime.now()
        records.append(high_quality_inferior_user)

    session.add_all(records) 
    session.commit()

    for uid in all_user_tag_df.index:
        score = all_user_tag_df.loc[uid, 'score']
        service_score = all_user_tag_df.loc[uid, 'service_score']
        session.query(asset_allocation.user_account_infos).filter(asset_allocation.user_account_infos.ua_uid == uid).update({"ua_uq_score" : score, "ua_uqs_score" : service_score})
        print(uid)
        session.commit()


    session.commit()
    session.close()


@gb.command()
@click.pass_context
def test(ctx):
    all_user_tag_df = pd.read_csv('./data/high_inferior_user_score.csv', index_col = ['uid'])
    haiyong_user_df = pd.read_csv('./data/haiyong_user.csv')
    good = haiyong_user_df.good.dropna()
    bad = haiyong_user_df.bad.dropna()

    uids = all_user_tag_df.index & good
    #print(all_user_tag_df.loc[uids].mean(), all_user_tag_df.loc[uids].std())

    uids = all_user_tag_df.index & bad
    bad_user_score = all_user_tag_df.loc[uids]
    bad_user_score = bad_user_score.sort_values(by = ['score'], ascending = False)
    bad_user_score.index = bad_user_score.index.astype(str)
    #print(bad_user_score)
    #print(all_user_tag_df.loc[uids].mean(), all_user_tag_df.loc[uids].std())
    #print(all_user_tag_df.loc[uids].mean(), all_user_tag_df.loc[uids].std())
    
    uids = all_user_tag_df.index & good
    good_user_score = all_user_tag_df.loc[uids]
    good_user_score = good_user_score.sort_values(by = ['score'], ascending = False)
    good_user_score.index = good_user_score.index.astype(str)
    print(good_user_score)

