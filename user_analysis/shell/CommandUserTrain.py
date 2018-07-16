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
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model


import traceback, code

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def train(ctx):

    '''user analysis
    '''
    pass



@train.command()
@click.pass_context
def question_risk_xgboost(ctx):

    risk_positive_negative_df = pd.read_csv('tmp/risk_positive_negative.csv', index_col = ['risk'])
    user_question_feature_df = pd.read_csv('tmp/user_question_answer_feature.csv', index_col = ['uq_uid'])


    positive_negative = risk_positive_negative_df.loc[1.0]

    positive = positive_negative.loc['positive']
    positive_feature = user_question_feature_df.loc[json.loads(positive)].dropna().drop_duplicates()

    negative = positive_negative.loc['negative']
    negative_feature = user_question_feature_df.loc[json.loads(negative)].dropna().drop_duplicates()

    feature = pd.concat([positive_feature, negative_feature] ,axis = 0)
    label = np.append(np.ones(len(positive_feature)), np.zeros(len(negative_feature)))

    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=43)

    dtrain = xgb.DMatrix(X_train, label = y_train)
    dval = xgb.DMatrix(X_val, label = y_val)
    dtest = xgb.DMatrix(X_test, label = y_test)

    # specify parameters via map
    param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
    param['nthread'] = 16
    param['eval_metric'] = 'auc'
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    #evallist = [(dtrain, 'train')]
    num_round = 1000000
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=500)
    #bst = xgb.cv(param, dtrain, num_round, nfold = 4, early_stopping_rounds=500)
    # make prediction
    preds = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    print(roc_auc_score(y_test, preds))


    #threshold = 0.3
    #preds[preds >= threshold] = 1.0
    #preds[preds < threshold] = 0.0

    #print(preds)
    #print(accuracy_score(y_test, preds))
    #print(recall_score(y_test, preds))
    #print(confusion_matrix(y_test, preds))
    #print(len(label == 0))
    #print(len(preds == 0))


    logistic = linear_model.LogisticRegression()
    logistic.fit(X_train, y_train)
    preds = logistic.predict(X_test)
    print(roc_auc_score(y_test, preds))
    threshold = 0.5
    preds[preds >= threshold] = 1.0
    preds[preds < threshold] = 0.0
    #print(accuracy_score(y_test, preds))
    #print(recall_score(y_test, preds))
    #print(confusion_matrix(y_test, preds))
