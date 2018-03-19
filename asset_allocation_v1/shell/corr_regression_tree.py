#coding=utf8


import getopt
import string
import json
import os
import sys
import logging
sys.path.append('shell')
import click
import config
import pandas as pd
import numpy as np
import time
import DBData


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund
from db.asset_stock_factor import *
from db.asset_stock import *
from db import asset_trade_dates, asset_ra_pool
from multiprocessing import Pool
import math
import scipy.stats as stats
import json
import statsmodels.api as sm
import statsmodels
import time
import functools
import Portfolio as PF


from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.tree import DecisionTreeRegressor


import stock_util
import stock_factor_util


logger = logging.getLogger(__name__)


def regression_tree_factor_corr_layer(bf_ids):


    all_stocks = stock_util.all_stock_info()
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_exposure.trade_date, barra_stock_factor_exposure.stock_id, barra_stock_factor_exposure.bf_id ,barra_stock_factor_exposure.factor_exposure).filter(and_(barra_stock_factor_exposure.bf_id.in_(bf_ids), barra_stock_factor_exposure.trade_date.in_(stock_factor_util.month_last_day()))).statement
    factor_df = pd.read_sql(sql, session.bind , index_col = ['trade_date', 'stock_id', 'bf_id'], parse_dates = ['trade_date'])

    factor_df.to_csv('barra_stock_factor/factor_exposure.csv')

    #print factor_df.tail()

    session.commit()
    session.close()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate ,tq_sk_yieldindic.secode, tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement
    yield_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']) / 100.0
    yield_df = yield_df.unstack()
    yield_df.columns = yield_df.columns.droplevel(0)
    yield_df = yield_df.rename(columns = secode_globalid_dict)

    yield_df.to_csv('barra_stock_factor/yield.csv')

    #print yield_df.tail()
    session.commit()
    session.close()

    #factor_df = pd.read_csv('barra_stock_factor/factor_exposure.csv', index_col = ['trade_date', 'stock_id', 'bf_id'], parse_dates = ['trade_date'])
    #yield_df = pd.read_csv('barra_stock_factor/yield.csv', index_col = ['tradedate'], parse_dates = ['tradedate'])
    factor_df = factor_df[factor_df.index.get_level_values(0) >= '2003-01-01']

    dates = list(set(factor_df.index.get_level_values(0).ravel()))
    dates.sort()

    params = {}
    params['min_leaf_node'] = 5
    #params['threads'] = 30


    for date in dates[15:]:
        tmp_yield_df = yield_df[yield_df.index <= date]
        tmp_yield_df = tmp_yield_df.iloc[-252:,]
        tmp_yield_df = tmp_yield_df.dropna(axis = 1, thresh = 200).fillna(0.0)
        #财会数据库有错误数据,股票每天的涨跌幅不变，没有波动,没法计算相关系数,需要更改
        stocks = tmp_yield_df.columns[tmp_yield_df.std() > 0].ravel()
        tmp_yield_df = tmp_yield_df[stocks]
        tmp_factor = factor_df.loc[date]
        tmp_factor = tmp_factor.unstack()
        tmp_factor.columns = tmp_factor.columns.droplevel(0)
        stocks = list(tmp_yield_df.columns & tmp_factor.index)
        #stocks = stocks[0:200]
        tmp_yield_df = tmp_yield_df[stocks]
        tmp_factor = tmp_factor.loc[stocks]
        tmp_factor = tmp_factor.dropna(axis = 1, thresh = len(tmp_factor) * 0.8).dropna()
        stocks = tmp_factor.index.ravel()
        tmp_yield_df = tmp_yield_df[stocks]
        corr_df = tmp_yield_df.corr()


        tree = create_regression_tree(tmp_factor.copy(), corr_df, 6, params)

        engine = database.connection('asset')
        Session = sessionmaker(bind=engine)
        session = Session()

        #record = session.query(barra_stock_factor_regression_tree.tree).filter(barra_stock_factor_regression_tree.trade_date == date).first()
        #tree = json.loads(record[0])

        clusters = {}
        for stock in tmp_factor.index:
            stock_feature = tmp_factor.loc[stock]
            name = ''
            tmp_tree = tree
            while True:
                if tmp_tree is None:
                    cluster = clusters.setdefault(name, [])
                    cluster.append(stock)
                    break
                else:
                    bf_feature = tmp_tree['feature']
                    bf_value = tmp_tree['value']
                    if stock_feature.loc[bf_feature] < bf_value:
                        tmp_tree = tmp_tree['left']
                        name = name + '_' + str(bf_feature) + '_left'
                    if stock_feature.loc[bf_feature] >= bf_value:
                        tmp_tree = tmp_tree['right']
                        name = name + '_' + str(bf_feature) + '_right'

        print pd.to_datetime(date).strftime('%Y-%m-%d')
        keys = list(clusters.keys())
        keys.sort()
        for key in keys:
            print key, clusters[key]


        bsfrt = barra_stock_factor_regression_tree()
        bsfrt.trade_date = date
        bsfrt.tree = json.dumps(tree)
        bsfrt.clusters = json.dumps(clusters)
        session.merge(bsfrt)


        session.commit()
        session.close()

        #print pd.to_datetime(date).strftime('%Y-%m-%d') ,tree
        #corr_df = corr_df[corr_df > 0.9]
        #print tmp_yield_df.corr().dropna(axis = 0)
        #tmp_yield_corr_df = tmp_yield_df.corr().dropna()
        #stocks = tmp_yield_corr_df.index.ravel()
        #tmp_factor = tmp_factor.loc[stocks]
        #print tmp_factor

        #print tmp_yield_df.corr()
        #print date , stocks
        #tmp_factor = tmp_factor.dropna()
        #print tmp_factor
    #create_regression_tree()

    return


#计算误差, 误差为相关性均值除以相关性波动率
def err(stock_ids, corr_df):

    if len(stock_ids) == 0:
        return -np.inf

    tmp_corr_df = corr_df.ix[stock_ids, stock_ids]
    corrs = tmp_corr_df.values.reshape(-1)
    corrs = corrs[corrs < 1.0]

    return corrs.mean()



def m_best_feature_value(stock_feature_df, corr_df, params, bf_id):

    best_err, best_feature, best_value, best_left_err, best_right_err  = -np.inf, None, None, None, None

    feature = stock_feature_df[bf_id]
    feature = feature.sort_values()

    if len(feature) <= params['min_leaf_node']:
        print best_feature, best_value, best_err, len(feature)


    for i in range(params['min_leaf_node'], len(feature) - params['min_leaf_node']):
        v = feature.iloc[i]
        left_err = err(feature[feature < v].index.ravel(), corr_df)
        right_err = err(feature[feature >= v].index.ravel(), corr_df)
        left_len = len(feature[feature < v])
        lr_err = 1.0 * left_len / len(feature) * left_err + (1 - 1.0 * left_len / len(feature)) * right_err
        if lr_err > best_err:
            best_err = lr_err
            best_feature = bf_id
            best_value = v
            best_left_err = left_err
            best_right_err = right_err

    print best_feature, best_value, best_err, len(feature), len(feature[feature < best_value]), len(feature[feature >= best_value]), best_left_err, best_right_err
    return best_feature, best_value, best_err


#选择分支
def choose_best_feature_value(stock_feature_df, corr_df, params):


    best_err, best_feature, best_value  = -np.inf, None, None

    pool = Pool(len(stock_feature_df.columns))
    records = pool.map(functools.partial(m_best_feature_value, stock_feature_df, corr_df, params), stock_feature_df.columns)
    pool.close()
    pool.join()

    for record in records:
        bf_id = record[0]
        value = record[1]
        lrerr = record[2]
        if lrerr > best_err:
            best_err = lrerr
            best_feature = bf_id
            best_value = value

    print
    print best_feature, best_value, best_err
    print

    return best_feature, best_value, best_err


def split_stocks(bf_id, value, stock_feature_df):

        feature = stock_feature_df[bf_id]

        left_stocks = feature[feature < value].index.ravel()
        right_stocks = feature[feature >= value].index.ravel()

        return stock_feature_df.loc[left_stocks], stock_feature_df.loc[right_stocks]


def create_regression_tree(stock_feature_df, corr_df, height, params):

    if height == 0:
        return None

    bf_id, bf_value, feature_err = choose_best_feature_value(stock_feature_df, corr_df, params)

    if bf_id is None or bf_value is None:
        return None

    else:
        left_stock_feature_df, right_stock_feature_df = split_stocks(bf_id, bf_value, stock_feature_df)
        tree = {}
        tree['feature'] = bf_id
        tree['value'] = bf_value
        tree['left'] = create_regression_tree(left_stock_feature_df, corr_df, height - 1, params)
        tree['right'] = create_regression_tree(right_stock_feature_df, corr_df, height - 1, params)

        return tree




def regression_tree_factor_cluster_boot(bf_ids):


    all_stocks = stock_util.all_stock_info()
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))


    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate ,tq_sk_yieldindic.secode, tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement
    yield_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']).replace(0.0, np.nan) / 100.0
    yield_df = yield_df.unstack()
    yield_df.columns = yield_df.columns.droplevel(0)
    yield_df = yield_df.rename(columns = secode_globalid_dict)


    session.commit()
    session.close()


    #yield_df = pd.read_csv('barra_stock_factor/yield.csv', index_col = ['tradedate'], parse_dates = ['tradedate'])

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()


    sql = session.query(barra_stock_factor_regression_tree.trade_date, barra_stock_factor_regression_tree.clusters).statement
    cluster_df = pd.read_sql(sql, session.bind, index_col = ['trade_date'], parse_dates = ['trade_date'])
    session.commit()

    dates = DBData.trade_date_index(cluster_df.index[0], end_date=cluster_df.index[-1])
    all_stocks_globalids = list(set(all_stocks.globalid) & set(yield_df.columns))
    stock_pos_df = pd.DataFrame(0, index = dates, columns = all_stocks_globalids)


    for date in dates:
        factor_index_data = {}
        factor_stocks = {}
        tmp_cluster_df = cluster_df[cluster_df.index < date]
        if len(tmp_cluster_df) == 0:
            continue
        clusters = json.loads(tmp_cluster_df.iloc[-1].values[0])
        keys = list(clusters.keys())
        keys.sort()

        for key in keys:
            stocks = clusters[key]
            factor_stocks[key] = stocks

            pos = pd.DataFrame(1.0 / len(stocks), index = [date], columns = stocks)
            tmp_yield_df = yield_df[stocks]
            tmp_yield_df = tmp_yield_df[tmp_yield_df.index <= date]
            tmp_yield_df = tmp_yield_df.iloc[-120:,]
            pos = pos.reindex(tmp_yield_df.index).fillna(method = 'bfill')
            nav_df = (tmp_yield_df * pos).sum(axis = 1).dropna()
            nav_df = ( nav_df + 1 ).cumprod()
            factor_index_data[key] = nav_df


        factor_index_df = pd.DataFrame(factor_index_data)
        #print factor_index_df.tail()
        df_inc = factor_index_df.pct_change().fillna(0.0)

        df_inc = df_inc.iloc[-120:,]

        bound = []
        for asset in df_inc.columns:
            bound.append({'sum1': 0,    'sum2' : 0,   'upper': 1.0,  'lower': 0.0})
        risk, returns, ws, sharpe = PF.markowitz_bootstrape(df_inc, bound, cpu_count=32, bootstrap_count=0)

        cols = df_inc.columns[ws >= 0.05]
        df_inc = df_inc[cols]

        bound = []
        for asset in df_inc.columns:
            bound.append({'sum1': 0,    'sum2' : 0,   'upper': 1.0,  'lower': 0.0})
        risk, returns, ws, sharpe = PF.markowitz_bootstrape(df_inc, bound, cpu_count=32, bootstrap_count=0)

        print date , df_inc.columns, ws

        record = pd.Series(0, index = all_stocks.globalid)
        for i in range(0, len(ws)):
            #tmp_record = pd.Series(0, index = all_stocks.globalid)
            cluster_name = df_inc.columns[i]
            stocks = factor_stocks[cluster_name]
            stocks = list(set(stocks) & set(all_stocks.globalid.ravel()))
            w = ws[i]
            for stock_id in stocks:
                record.loc[stock_id] = record.loc[stock_id] + w * 1.0 / len(stocks)

        stock_pos_df.loc[date] = record

    return stock_pos_df



if __name__ == '__main__':

    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012',
            'BF.000013', 'BF.000014', 'BF.000015', 'BF.000016', 'BF.000017']

    #regression_tree_factor_corr_layer(bf_ids)
    regression_tree_factor_cluster_boot(bf_ids)

