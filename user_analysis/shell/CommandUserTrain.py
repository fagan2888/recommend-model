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
import warnings
warnings.filterwarnings("ignore")

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import LSTMBuyRedeem
import random
from torch.nn import utils as nn_utils
from torch.autograd import Variable


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


    positive_negative = risk_positive_negative_df.loc[0.4]

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



@train.command()
@click.pass_context
def user_redeem_lstm(ctx):

    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[32]')
            .set("spark.executor.memory", "50G")
            .set('spark.driver.memory', '50G')
            .set('spark.driver.maxResultSize', '50G'))

    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    feature = spark.read.csv("tmp/feature.csv", header = True)
    feature_rdd = feature.rdd.repartition(1000)

    def user_feature(x):

        uid = x[0]
        user_feature = pd.DataFrame([v.asDict() for v in x[1]]).fillna(np.nan)
        user_feature = user_feature.set_index(['ts_holding_uid', 'ts_date'])
        user_feature = user_feature.sort_index()
        user_feature = user_feature[['36', '46', 'ts_nav']]
        user_feature['36'] = user_feature['36'].astype(float)
        user_feature['46'] = user_feature['46'].astype(float)
        user_feature['ts_nav'] = user_feature['ts_nav'].astype(float)
        tag = np.zeros(len(user_feature))
        tag[user_feature['36'] == 1.0] = 1
        tag[user_feature['46'] == 1.0] = 2
        user_feature['inc'] = user_feature.ts_nav.pct_change().fillna(0.0)
        return [user_feature[['ts_nav', 'inc']].values.astype(float) ,tag.astype(int)]

    feature = feature_rdd.groupBy(lambda x : x.ts_holding_uid).map(user_feature).collect()

    print('feature num', len(feature))

    feature = sorted(feature, key = lambda x : len(x[0]), reverse = True)

    max_length = len(feature[0][0])

    batch_size = 50
    feature_dim = 2
    seq_length = [max_length, feature_dim]
    train = []
    for n in range(0, len(feature), batch_size):
        batch_feature = feature[n : min(n + batch_size, len(feature))]
        max_length_batch_feature = []
        max_length_batch_label = []
        seq_length = []
        for user in batch_feature:
            user_feature = user[0]
            user_label = user[1]
            feature_dim = user_feature.shape[1]
            seq_length.append(len(user_feature))
            npi = np.zeros(feature_dim, dtype=np.float32)
            npi = np.tile(npi, (max_length - len(user_feature), 1))
            user_feature = np.row_stack((user_feature, npi))
            npo = np.zeros(max_length - len(user_label), dtype=np.int)
            user_label = np.append(user_label, npo)
            max_length_batch_feature.append(user_feature)
            max_length_batch_label.append(user_label)
        #train.append([torch.tensor(user[0], dtype = torch.float).cuda(), torch.tensor(user[1], dtype = torch.long).cuda()])

        batch_feature_torch = torch.tensor(max_length_batch_feature , dtype = torch.float).cuda()
        batch_label_torch = torch.tensor(max_length_batch_label , dtype = torch.float).cuda()
        batch_feature_torch = nn_utils.rnn.pack_padded_sequence(batch_feature_torch, seq_length, batch_first=True)
        #batch_label_torch = nn_utils.rnn.pack_padded_sequence(batch_label_torch, seq_length, batch_first=True)
        train.append([batch_feature_torch, batch_label_torch])

    print('train num', len(train))

    hidden_dim = 3
    target_dim = 3

    model = LSTMBuyRedeem(feature_dim, hidden_dim, target_dim)
    model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10000):  # again, normally you would NOT do 300 epochs, it is toy data

        for inputs, targets in train:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.

            # Step 3. Run our forward pass.
            tag_scores = model(inputs)
            set_trace()
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)

            loss.backward()
            optimizer.step()


        #with torch.no_grad():
        #    inputs = torch.tensor(train[10][0], dtype = torch.float).cuda()
        #    tag_scores = model(inputs)
        #    print(tag_scores.argmax(1))
        #    print(train[10][1])

        print(epoch)


    torch.save(model.state_dict(), 'redeem_lstm_model')
