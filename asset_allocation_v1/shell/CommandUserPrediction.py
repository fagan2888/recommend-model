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
import xgboost as xgb
from sklearn.model_selection import train_test_split
import lightgbm as lgb


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
    holding_df = pd.read_csv('user_prediction/ts_holding_nav.csv', parse_dates = ['ts_date'])
    order_df = pd.read_csv('user_prediction/ts_order.csv', parse_dates = ['ts_placed_date'])

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
    uids = set()
    for uid, holding_group in holding_df.groupby(holding_df.index):

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
        #print trade_type3_df.head()


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
                #print uid
                #print feat_date_list
                #print buy_date
                buy_index = feat_date_list.index(buy_date)
                start_index = max(0, buy_index - 7)
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

        feat['risk_180_mean'] = feat['risk'].rolling(180).mean()
        feat['risk_90_mean'] = feat['risk'].rolling(90).mean()
        feat['risk_60_mean'] = feat['risk'].rolling(60).mean()
        feat['risk_30_mean'] = feat['risk'].rolling(30).mean()
        feat['risk_15_mean'] = feat['risk'].rolling(15).mean()
        feat['risk_7_mean'] = feat['risk'].rolling(7).mean()


        #holding_group.index.name = 'date'
        #order_group.index.name = 'date'
        #print holding_group.index
        #print order_group.index
        #print feat.columns

        #print holding_order_df.head()
        feats.append(feat)

    feat_df = pd.concat(feats, axis = 0)
    feat_df.index.name = 'ts_date'

    #print len(feat_df)

    feat_df.to_csv('./user_prediction/feat.csv')



@user.command()
@click.option('--featurefile', 'optfeaturefile', default=True, help=u'feature file path')
@click.pass_context
def xgboost(ctx, optfeaturefile):

    feat_df = pd.read_csv(optfeaturefile.strip(), index_col = ['ts_date'], parse_dates = ['ts_date'])

    train_df = feat_df[feat_df.index <= '2017-10-31']
    test_df  = feat_df[feat_df.index > '2017-10-31']

    y = train_df['label'].ravel()
    X = train_df.drop(['label'], axis = 1).values

    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state = 13243)

    test_X = test_df.drop(['label'], axis = 1).values
    test_y = test_df['label'].ravel()


    xg_train = xgb.DMatrix(train_X, label=train_y, missing = np.nan)
    xg_val = xgb.DMatrix(val_X, label=val_y, missing = np.nan)
    xg_test = xgb.DMatrix(test_X,  missing = np.nan)


    params = {}
    # use softmax multi-class classification
    params['objective'] = 'multi:softmax'
    # scale weight of positive examples
    params['eta'] = 0.1
    params['max_depth'] = 10
    #params['silent'] = 1
    params['nthread'] = 60
    params['num_class'] = 3
    params['eval_metric'] = 'mlogloss'
    #params['min_child_weight'] = 3
    #params['lambda'] = 100
    #params['gamma'] = 0.1
    #params['subsample'] = 1
    #params['colsample_bytree'] = 1
    #params['rate_drop'] = 0.1
    #params['skip_drop'] = 0.5
    #params['scale_pos_weight'] = float(np.sum(train_y == 0)) / np.sum(train_y==1)
    params['seed'] = 103


    watchlist = [(xg_train,'train'), (xg_val,'val')]
    num_round = 1000
    clf = xgb.train(params, xg_train, num_round, watchlist, early_stopping_rounds = 50)


    # get prediction
    pred = clf.predict(xg_test)
    print np.sum(pred)
    print np.sum(test_y)
    print pred
    error_rate = 1.0 * np.sum(pred != test_y) / test_y.shape[0]
    print('Test error using softmax = {}'.format(error_rate))


@user.command()
@click.option('--featurefile', 'optfeaturefile', default=True, help=u'feature file path')
@click.pass_context
def lightgbm(ctx, optfeaturefile):

    feat_df = pd.read_csv(optfeaturefile.strip(), index_col = ['ts_date'], parse_dates = ['ts_date'])

    train_df = feat_df[feat_df.index <= '2017-10-31']
    test_df  = feat_df[feat_df.index > '2017-10-31']

    train_y = train_df['label'].ravel()
    train_X = train_df.drop(['label'], axis = 1).values
    test_y = test_df['label'].ravel()
    test_X = test_df.drop(['label'], axis = 1).values

    #train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state = 13243)

    lgb_train = lgb.Dataset(train_X, train_y, free_raw_data=False)
    lgb_test = lgb.Dataset(test_X, test_y, reference=lgb_train, free_raw_data=False)

    params = {
            'task': 'train',
            'boosting_type': 'dart',
            'objective': 'binary',
            #'metric': {'auc'},
            'learning_rate': 0.1,
            'num_leaves': 112,
            'max_depth': 9,
            'min_data_in_leaf': 82,
            'min_sum_hessian_in_leaf': 1e-3,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_gain_to_split':0,
            #'lambda_l1': 50,
            #'lambda_l2': 130,
            'max_bin': 255,
            'drop_rate': 0.1,
            'skip_drop': 0.5,
            #'max_drop': 50,
            'nthread': 64,
            'verbose': -1,
            #'scale_pos_weight':weight
            }


    cvresult = lgb.cv(params, lgb_train, num_boost_round=1000, nfold=5, metrics='auc',
                    early_stopping_rounds=50, seed=123,verbose_eval=True)
    best_round = len(cvresult['auc-mean'])
    print 'best_num_boost_round: %d' % best_round
    gbm = lgb.train(params, lgb_train, num_boost_round=best_round, verbose_eval=False,valid_sets=[lgb_train,lgb_test])
    train_pre = gbm.predict(train_X)
    test_pre = gbm.predict(test_X)
    print "\nModel Report"
    print "AUC Score(Train): %f" % roc_auc_score(train_y, train_pre)
    print "AUC Score(Test) : %f" % roc_auc_score(test_y, test_pre)
    #return roc_auc_score(test_y, test_pre)
    return gbm




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
