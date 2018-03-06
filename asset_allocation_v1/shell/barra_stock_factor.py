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


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund
from db.asset_stock_factor import *
from db.asset_stock import *
from db import asset_trade_dates
from multiprocessing import Pool
import math
import scipy.stats as stats
import json
import statsmodels.api as sm
import statsmodels
import Portfolio as PF

from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.tree import DecisionTreeRegressor

import stock_util
import stock_factor_util


logger = logging.getLogger(__name__)



#回归计算因子收益率
def factor_yield(factor1_df, factor2_df):

    all_stocks = stock_util.all_stock_info()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate ,tq_sk_yieldindic.secode, tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement
    yield_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']).replace(0.0, np.nan) / 100.0
    yield_df = yield_df.unstack()
    yield_df.columns = yield_df.columns.droplevel(0)
    #yield_df.to_csv('barra_stock_factor/yield.csv')

    session.commit()
    session.close()

    #yield_df = pd.read_csv('barra_stock_factor/yield.csv', index_col = ['tradedate'], parse_dates = ['tradedate'])
    yield_df = yield_df[yield_df.index >= '2010-01-01']

    secodes = factor2_df.columns & factor1_df.columns & yield_df.columns
    factor1_df = factor1_df[secodes]
    factor2_df = factor2_df[secodes]
    yield_df = yield_df[secodes]

    dates = factor2_df.index & factor1_df.index & yield_df.index
    factor_yield = []
    factor_yield_dates = []
    for date in dates:
        factor1 = factor1_df.loc[date]
        factor2 = factor2_df.loc[date]
        Yield = yield_df.loc[date]
        df = pd.DataFrame({'exposure1':factor1, 'exposure2':factor2, 'yield':Yield}).dropna()

        if len(df) <= 30:
            continue

        X = df[['exposure1','exposure2']].values
        X = sm.add_constant(X)
        y = df['yield'].ravel()

        #print len(X)

        #if date.strftime('%Y-%m-%d') == '2017-05-04':
        #    df.to_csv('df.csv')
        #    sys.exit(0)

        result = sm.OLS(y, X).fit()
        print date, result.params
        factor_r = result.params[1:]
        factor_yield_dates.append(date)
        factor_yield.append(factor_r)

    factor_yield_df = pd.DataFrame(factor_yield, index = factor_yield_dates, columns = ['yield1', 'yield2'])
    factor_yield_df = factor_yield_df.loc[dates].fillna(0.0)
    #pe_yield_df = (pe_yield_df + 1).cumprod()

    factor_yield_df.to_csv('factor_yield.csv')
    factor_nav_df = (factor_yield_df + 1).cumprod()
    factor_nav_df.to_csv('factor_nav.csv')



#多因子暴露聚类
def multi_factor_kmeans(factor_df):


    all_stocks = stock_util.all_stock_info()
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    factor_df = factor_df[factor_df.index.get_level_values(0) >= '2017-01-01']

    dates = list(set(factor_df.index.get_level_values(0)))
    dates.sort()

    for date in dates:

        factors = factor_df.loc[date]
        factors = factors.dropna()
        if len(factors) <= 30:
            continue

        X = factors.values

        #print X.shape
        kmeans = KMeans(n_clusters=10, random_state=23).fit(X)
        #print len(kmeans.labels_)
        nums = {}
        for label in set(kmeans.labels_):
            print factors[kmeans.labels_ == label].mean(axis = 0)
            #nums[factors[kmeans.labels_ == label].mean()] = factors[kmeans.labels_ == label].index
            #nums[factors[kmeans.labels_ == label].mean()] = len(factors[kmeans.labels_ == label].index)

        nums = sorted(nums.items(), reverse = True)
        #print date, nums
        nums = [x[1] for x in nums]
        #print date, nums
        layer_stocks = nums[2].ravel()

        print date, len(layer_stocks)
        stock_pos[date] = layer_stocks

    stock_pos_df = stock_util.stock_pos_2_weight(stock_pos)

    return stock_pos_df


#按照聚类确定分层
def factor_layer_stocks(bf_id):

    cluster_num = 10

    all_stocks = stock_util.all_stock_info()
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    #导出factor exposure
    sql = session.query(barra_stock_factor_exposure.trade_date, barra_stock_factor_exposure.stock_id, barra_stock_factor_exposure.factor_exposure).filter(barra_stock_factor_exposure.bf_id == bf_id).statement
    factor_df = pd.read_sql(sql, session.bind , index_col = ['trade_date', 'stock_id'], parse_dates = ['trade_date'])
    factor_df = factor_df.unstack()
    factor_df.columns = factor_df.columns.droplevel(0)

    factor_df = factor_df[factor_df.index >= '2005-01-01']

    for i in range(0, len(factor_df.index)):

        date = factor_df.index[i]

        factors = factor_df.loc[date]
        factors = factors.dropna()
        if len(factors) <= 30:
            continue

        X = factors.ravel().reshape(-1, 1)

        #根据因子暴露度聚为5个类
        kmeans = KMeans(n_clusters=cluster_num, random_state=23).fit(X)
        nums = {}
        for label in set(kmeans.labels_):
            nums[factors[kmeans.labels_ == label].mean()] = factors[kmeans.labels_ == label].index

        nums = sorted(nums.items(), reverse = True)
        nums = [x[1] for x in nums]

        for j in range(0, cluster_num):
            layer_stocks = nums[j].ravel()
            bsfys = barra_stock_factor_layer_stocks()
            bsfys.bf_id = bf_id
            bsfys.trade_date = date
            bsfys.layer = j
            bsfys.stock_ids = json.dumps(list(layer_stocks))
            session.merge(bsfys)

        session.commit()

        print bf_id, date

    session.close()



#利用回归树对因子分层
def regression_tree_factor_layer(bf_id):


    all_stocks = stock_util.all_stock_info()
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    #导出factor exposure
    sql = session.query(barra_stock_factor_exposure.trade_date, barra_stock_factor_exposure.stock_id ,barra_stock_factor_exposure.factor_exposure).filter(barra_stock_factor_exposure.bf_id == bf_id).filter(barra_stock_factor_exposure.trade_date.in_(stock_factor_util.month_last_day())).statement
    factor_df = pd.read_sql(sql, session.bind , index_col = ['trade_date', 'stock_id'], parse_dates = ['trade_date'])

    factor_df = factor_df.unstack()
    factor_df.columns = factor_df.columns.droplevel(0)

    session.commit()
    session.close()


    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate ,tq_sk_yieldindic.secode, tq_sk_yieldindic.yieldm).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement
    yield_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']) / 100.0
    yield_df = yield_df.unstack()
    yield_df.columns = yield_df.columns.droplevel(0)
    yield_df = yield_df.rename(columns = secode_globalid_dict)

    session.commit()
    session.close()



    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()


    dates = factor_df.index & yield_df.index & stock_factor_util.month_last_day()
    look_back = 6

    stock_pos = {}
    for i in range(look_back + 1, len(dates) - 1):

        date = dates[i]
        print bf_id, date
        factor_dates = dates[i - look_back - 1 : i]
        yield_dates  = dates[i - look_back : i + 1]
        tmp_factor_df = factor_df.loc[factor_dates]
        tmp_yield_df = yield_df.loc[yield_dates]

        X = []
        y = []
        all_stocks = []
        for j in range(0, len(factor_dates)):
            factors = tmp_factor_df.loc[factor_dates[j]]
            yields = tmp_yield_df.loc[yield_dates[j]]
            factors = factors.dropna()
            yields = yields.dropna()
            stocks = factors.index & yields.index
            #print factors
            for stock in stocks:
                X.append([factors.loc[stock]])
                y.append(yields.loc[stock])
                all_stocks.append(stock)

        if len(X) <= 1000:
            continue

        X = np.array(X)
        regr = DecisionTreeRegressor(max_depth = 3, min_samples_leaf  = max(len(X) / look_back / 50 * 4, 20))
        regr.fit(X, y)
        predicts = regr.predict(X)

        value_stocks = {}
        for p in range(0, len(predicts)):
           pred = predicts[p]
           stock = all_stocks[p]
           stocks = value_stocks.setdefault(pred, set())
           stocks.add(stock)

        exposure_stocks = {}
        for stocks in value_stocks.values():
            stocks = list(stocks)
            exposure_stocks[factor_df.loc[date, stocks].mean()] = stocks

        exposures = exposure_stocks.keys()
        exposures.sort()
        for i in range(0 ,len(exposures)):
            stocks = exposure_stocks[exposures[i]]
            bsfys = barra_stock_factor_layer_stocks()
            bsfys.bf_id = bf_id
            bsfys.trade_date = date
            bsfys.layer = i
            bsfys.stock_ids = json.dumps(list(stocks))
            session.merge(bsfys)


    session.commit()
    session.close()

    return


def regression_tree_factor_spliter(bf_ids):

    all_stocks = stock_util.all_stock_info()
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    #导出factor exposure
    sql = session.query(barra_stock_factor_exposure.trade_date, barra_stock_factor_exposure.stock_id, barra_stock_factor_exposure.bf_id ,barra_stock_factor_exposure.factor_exposure).filter(and_(barra_stock_factor_exposure.bf_id.in_(bf_ids), barra_stock_factor_exposure.trade_date.in_(stock_factor_util.month_last_day()))).statement
    factor_df = pd.read_sql(sql, session.bind , index_col = ['trade_date', 'stock_id', 'bf_id'], parse_dates = ['trade_date'])

    #factor_df = factor_df.unstack(level = [1,2])
    #factor_df.columns = factor_df.columns.droplevel(0)
    factor_df.to_csv('barra_stock_factor/factor_exposure.csv')

    session.commit()
    session.close()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate ,tq_sk_yieldindic.secode, tq_sk_yieldindic.yieldm).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement
    yield_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']) / 100.0
    yield_df = yield_df.unstack()
    yield_df.columns = yield_df.columns.droplevel(0)
    yield_df = yield_df.rename(columns = secode_globalid_dict)

    yield_df.to_csv('barra_stock_factor/yieldm.csv')

    print yield_df.tail()
    session.commit()
    session.close()


    look_back = 6

    factor_df = pd.read_csv('barra_stock_factor/factor_exposure.csv', index_col = ['trade_date', 'stock_id', 'bf_id'], parse_dates = ['trade_date'])
    factor_df = factor_df.unstack(level = [1,2])
    factor_df.columns = factor_df.columns.droplevel(0)
    factor_df = factor_df.dropna(thresh = 500)
    yield_df = pd.read_csv('barra_stock_factor/yieldm.csv' , index_col = ['tradedate'], parse_dates = ['tradedate'])

    dates = factor_df.index & yield_df.index & stock_factor_util.month_last_day()
    #dates = dates[-20:]
    factor_df = factor_df.loc[dates]
    yield_df = yield_df.loc[dates]
    yield_df = stock_factor_util.normalized(yield_df)


    stock_pos = {}
    for i in range(look_back + 1, len(dates) - 1):

        factor_dates = dates[i - look_back - 1 : i]
        yield_dates  = dates[i - look_back : i + 1]
        tmp_factor_df = factor_df.loc[factor_dates]
        tmp_yield_df = yield_df.loc[yield_dates]

        X = []
        y = []
        all_stocks = []
        for j in range(0, len(factor_dates)):
            factors = tmp_factor_df.loc[factor_dates[j]]
            yields = tmp_yield_df.loc[yield_dates[j]]
            factors = factors.dropna()
            yields = yields.dropna()
            factors = factors.unstack()
            stocks = factors.index & yields.index
            for stock in stocks:
                bf_factors = factors.loc[stock]
                stock_factors = []
                for bf_id in bf_ids:
                    if bf_id in set(bf_factors.index) and (not np.isnan(bf_factors.loc[bf_id])):
                        stock_factors.append(bf_factors.loc[bf_id])
                    else:
                        stock_factors.append(0.0)
                X.append(stock_factors)
                y.append(yields.loc[stock])
                all_stocks.append(stock)


        X = np.array(X)
        regr = DecisionTreeRegressor(max_depth = 5, min_samples_leaf  = max(len(X) / look_back / 50 * 8, 20))
        regr.fit(X, y)
        print dates[i], regr.feature_importances_
        #print dates[i], len(set(regr.predict(X)))
        predicts = regr.predict(X)

        value_stocks = {}
        for p in range(0, len(predicts)):
           pred = predicts[p]
           stock = all_stocks[p]
           stocks = value_stocks.setdefault(pred, set())
           stocks.add(stock)


        #key = max(value_stocks.keys())
        #stocks = value_stocks[key]
        #print dates[i], value_stocks.values()
        #stock_pos[dates[i]] = list(stocks)
        #for key in value_stocks.keys():
        #    print dates[i], key
        #    stocks = list(value_stocks[key]))
        ##score = regr.score(X, y)
        #print regr.tree_
        #print regr.decision_path(X)
        #print len(X)
        stock_pos[dates[i]] = value_stocks.values()

    #print stock_pos

    yield_df = pd.read_csv('barra_stock_factor/yield.csv' , index_col = ['tradedate'], parse_dates = ['tradedate'])
    all_stocks = stock_util.all_stock_info()

    dates = stock_pos.keys()
    dates = list(dates)
    dates.sort()
    stock_pos_df = pd.DataFrame(0, index = dates, columns = all_stocks.globalid)
    for date in dates:
        stocks_array = stock_pos[date]
        navs = []
        for stocks in stocks_array:
            stocks = list(stocks)
            #print date , stocks

            pos = pd.DataFrame(1.0 / len(stocks), index = [date], columns = stocks)
            tmp_yield_df = yield_df[stocks]
            #print pos
            tmp_yield_df = tmp_yield_df[tmp_yield_df.index <= date]
            tmp_yield_df = tmp_yield_df.iloc[-120:,]
            pos = pos.reindex(tmp_yield_df.index).fillna(method = 'bfill')
            nav_df = (tmp_yield_df * pos).sum(axis = 1).dropna()
            nav_df = ( nav_df + 1 ).cumprod()

            navs.append(nav_df)

        nav_df = pd.concat(navs, axis = 1)
        df_inc = nav_df.pct_change().fillna(0.0)

        bound = []
        for asset in df_inc.columns:
            bound.append({'sum1': 0,    'sum2' : 0,   'upper': 1.0,  'lower': 0.0})

        risk, returns, ws, sharpe = PF.markowitz_bootstrape(df_inc, bound, cpu_count=32, bootstrap_count=0)

        print date, ws

        record = pd.Series(0, index = all_stocks.globalid)
        for i in range(0, len(ws)):
            stocks = stocks_array[i]
            stocks = list(set(stocks) & set(all_stocks.globalid.ravel()))
            w = ws[i]
            for stock_id in stocks:
                record.loc[stock_id] = record.loc[stock_id] + w * 1.0 / len(stocks)
        stock_pos_df.loc[date] = record

    #print stock_pos_df.sum(axis = 1)
    return stock_pos_df



#计算各层净值
def factor_layer_nav(bf_id):

    all_stocks = stock_util.all_stock_info()
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))
    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_layer_stocks.trade_date, barra_stock_factor_layer_stocks.layer, barra_stock_factor_layer_stocks.stock_ids).filter(barra_stock_factor_layer_stocks.bf_id == bf_id).statement
    layer_stock_df = pd.read_sql(sql, session.bind , index_col = ['trade_date', 'layer'], parse_dates = ['trade_date'])

    session.commit()
    session.close()

    layer_stock_df = layer_stock_df.unstack()
    layer_stock_df.columns = layer_stock_df.columns.droplevel(0)

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate ,tq_sk_yieldindic.secode, tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement
    yield_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']) / 100.0
    yield_df = yield_df.unstack()
    yield_df.columns = yield_df.columns.droplevel(0)
    yield_df = yield_df.rename(columns = secode_globalid_dict)

    yield_df.to_csv('barra_stock_factor/yield.csv')

    #yield_df = pd.read_csv('barra_stock_factor/yield.csv', index_col = ['tradedate'], parse_dates = ['tradedate'])

    session.commit()
    session.close()

    layer_stock_df = layer_stock_df.loc[layer_stock_df.index & stock_factor_util.month_last_day()]


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    for layer in layer_stock_df.columns:
        stock_ser = layer_stock_df[layer]
        stock_pos = {}
        for date in stock_ser.index:
            stocks = json.loads(stock_ser.loc[date])
            stock_pos[date] = stocks

        stock_pos_df = stock_util.stock_pos_2_weight(stock_pos)

        stock_ids = yield_df.columns & stock_pos_df.columns
        tmp_yield_df = yield_df[stock_ids]
        stock_pos_df = stock_pos_df[stock_ids]

        stock_pos_df = stock_pos_df.reindex(tmp_yield_df.index).shift(1).fillna(method = 'pad')

        nav_df = (tmp_yield_df * stock_pos_df).sum(axis = 1).dropna()
        nav_df = ( nav_df + 1 ).cumprod()

        for date in nav_df.index:
            bsfln = barra_stock_factor_layer_nav()
            bsfln.bf_id = bf_id
            bsfln.layer = layer
            bsfln.trade_date = date
            bsfln.nav = nav_df.loc[date]
            session.merge(bsfln)

        print bf_id, layer, nav_df.tail()

        session.commit()

    session.commit()
    session.close()

    return


#计算各层加权净值
def factor_layer_weight_nav(bf_id):

    all_stocks = stock_util.all_stock_info()
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))
    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_layer_stocks.trade_date, barra_stock_factor_layer_stocks.layer, barra_stock_factor_layer_stocks.stock_ids).filter(barra_stock_factor_layer_stocks.bf_id == bf_id).statement
    layer_stock_df = pd.read_sql(sql, session.bind , index_col = ['trade_date', 'layer'], parse_dates = ['trade_date'])

    session.commit()
    session.close()

    layer_stock_df = layer_stock_df.unstack()
    layer_stock_df.columns = layer_stock_df.columns.droplevel(0)

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate ,tq_sk_yieldindic.secode, tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement
    yield_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']) / 100.0
    yield_df = yield_df.unstack()
    yield_df.columns = yield_df.columns.droplevel(0)
    yield_df = yield_df.rename(columns = secode_globalid_dict)

    session.commit()
    session.close()

    layer_stock_df = layer_stock_df.loc[layer_stock_df.index & stock_factor_util.month_last_day()]


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    for layer in layer_stock_df.columns:
        if layer == 1 or layer == 2 or layer == 3:
            continue
        stock_ser = layer_stock_df[layer]
        stock_pos = {}
        for date in stock_ser.index:
            stocks = json.loads(stock_ser.loc[date])
            stock_pos[date] = stocks

        stock_pos_df = stock_util.stock_pos_2_factor_weight(bf_id, stock_pos)

        stock_ids = yield_df.columns & stock_pos_df.columns
        tmp_yield_df = yield_df[stock_ids]
        stock_pos_df = stock_pos_df[stock_ids]

        stock_pos_df = stock_pos_df.reindex(tmp_yield_df.index).shift(1).fillna(method = 'pad')

        nav_df = (tmp_yield_df * stock_pos_df).sum(axis = 1).dropna()
        nav_df = ( nav_df + 1 ).cumprod()

        for date in nav_df.index:
            bsfln = barra_stock_factor_layer_weight_nav()
            bsfln.bf_id = bf_id
            bsfln.layer = layer
            bsfln.trade_date = date
            bsfln.nav = nav_df.loc[date]
            session.merge(bsfln)

        print bf_id, layer, nav_df.tail()

        session.commit()

    session.commit()
    session.close()

    return


#计算因子的分层IC值
def factor_layer_ic(bf_id):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_layer_nav.trade_date, barra_stock_factor_layer_nav.layer, barra_stock_factor_layer_nav.nav).filter(barra_stock_factor_layer_nav.bf_id == bf_id).statement
    layer_nav_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'layer'], parse_dates = ['trade_date'])
    layer_nav_df = layer_nav_df.unstack()
    layer_nav_df.columns = layer_nav_df.columns.droplevel(0)

    layer_nav_df = layer_nav_df.loc[layer_nav_df.index & stock_factor_util.month_last_day()]

    layer_nav_inc = layer_nav_df.pct_change().fillna(0.0)

    layer_nav_inc = layer_nav_inc.shift(-1).dropna()

    sql = session.query(barra_stock_factor_exposure.stock_id, barra_stock_factor_exposure.trade_date, barra_stock_factor_exposure.factor_exposure).filter(barra_stock_factor_exposure.bf_id == bf_id).statement
    all_factor_exposure_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'stock_id'])
    sql = session.query(barra_stock_factor_layer_stocks.layer, barra_stock_factor_layer_stocks.trade_date ,barra_stock_factor_layer_stocks.stock_ids).filter(barra_stock_factor_layer_stocks.bf_id == bf_id).statement
    all_factor_layer_stock_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'layer'])

    spearmanrs = []
    dates = []
    for date in layer_nav_inc.index:
        inc = layer_nav_inc.loc[date]
        #print bf_id, inc

        factor_exposure_df = all_factor_exposure_df.loc[date]
        factor_layer_stock_df = all_factor_layer_stock_df.loc[date]
        layer_factor_exposure = []
        layer_inc = []
        for layer in factor_layer_stock_df.index:
            layer_stocks = json.loads(factor_layer_stock_df.loc[layer].ravel()[0])
            layer_factor_exposure.append(factor_exposure_df.loc[layer_stocks].mean().values[0])
            layer_inc.append(inc.loc[layer])

        #print layer_factor_exposure, layer_inc

        #spearmanr = -1.0 * stats.stats.spearmanr(inc, inc.index).correlation
        spearmanr = np.corrcoef(layer_factor_exposure, layer_inc)[0][1]
        spearmanrs.append(spearmanr)
        dates.append(date)

    spearmanrs_df = pd.DataFrame(spearmanrs, index = dates, columns = [bf_id])

    for date in spearmanrs_df.index:
        bsfli = barra_stock_factor_layer_ic()
        bsfli.bf_id = bf_id
        bsfli.trade_date = date
        bsfli.ic = spearmanrs_df.loc[date, bf_id]
        session.merge(bsfli)

    session.commit()
    session.close()




#计算因子的分层IC值
def regression_tree_factor_layer_ic(bf_id):


    all_stocks = stock_util.all_stock_info()
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    #导出factor exposure
    sql = session.query(barra_stock_factor_exposure.trade_date, barra_stock_factor_exposure.stock_id ,barra_stock_factor_exposure.factor_exposure).filter(barra_stock_factor_exposure.bf_id == bf_id).filter(barra_stock_factor_exposure.trade_date.in_(stock_factor_util.month_last_day())).statement
    factor_df = pd.read_sql(sql, session.bind , index_col = ['trade_date', 'stock_id'], parse_dates = ['trade_date'])

    factor_df = factor_df.unstack()
    factor_df.columns = factor_df.columns.droplevel(0)

    session.commit()
    session.close()

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate ,tq_sk_yieldindic.secode, tq_sk_yieldindic.yieldm).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement
    yield_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']) / 100.0
    yield_df = yield_df.unstack()
    yield_df.columns = yield_df.columns.droplevel(0)
    yield_df = yield_df.rename(columns = secode_globalid_dict)

    session.commit()
    session.close()

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_layer_stocks.layer, barra_stock_factor_layer_stocks.trade_date ,barra_stock_factor_layer_stocks.stock_ids).filter(barra_stock_factor_layer_stocks.bf_id == bf_id).statement
    all_factor_layer_stock_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'layer'])


    all_factor_layer_stock_df = all_factor_layer_stock_df.unstack()
    all_factor_layer_stock_df.columns = all_factor_layer_stock_df.columns.droplevel(0)

    for date in all_factor_layer_stock_df.index:
        layer_stocks = all_factor_layer_stock_df.loc[date].dropna()
        layer_stocks = layer_stocks.sort_index(ascending=True)
        layer_exposure = []
        layer_yieldm = []
        for layer in layer_stocks.index:
            stocks = json.loads(layer_stocks[layer])
            exposure = factor_df.loc[date, stocks].mean()
            yieldm = yield_df.loc[date, stocks].mean()
            layer_exposure.append(exposure)
            layer_yieldm.append(yieldm)

        ic = np.corrcoef(layer_exposure, layer_yieldm)[0][1]
        print bf_id, date, ic

        bsfli = barra_stock_factor_layer_ic()
        bsfli.bf_id = bf_id
        bsfli.trade_date = date
        bsfli.ic = ic
        session.merge(bsfli)

    session.commit()
    session.close()

    return


#根据各层秩相关选取因子
def corr_factor_layer_selector(bf_ids):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_layer_ic.trade_date, barra_stock_factor_layer_ic.bf_id, barra_stock_factor_layer_ic.ic).filter(barra_stock_factor_layer_ic.bf_id.in_(bf_ids)).statement
    layer_ic_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'bf_id'], parse_dates = ['trade_date'])
    layer_ic_df = layer_ic_df.unstack()
    layer_ic_df.columns = layer_ic_df.columns.droplevel(0)

    layer_ic_df = layer_ic_df.rolling(6).mean()
    layer_ic_abs = abs(layer_ic_df)

    dates = []
    factor_layers = []
    for i in range(1, len(layer_ic_abs.index)):
        now_date = layer_ic_abs.index[i]
        date = layer_ic_abs.index[i - 1]
        ic = layer_ic_abs.loc[date]
        ic = ic.sort_values(ascending = False)
        date_factor_layer = []
        for bf_id in ic.index:
            ic_v = ic.loc[bf_id]
            if ic_v <= 0.3:
                continue
            if layer_ic_df.loc[date , bf_id] > 0:
                layer = 0
            else:
                layer = 9
            #date_factor_layer.append((bf_id , 0))
            #date_factor_layer.append((bf_id , 6))
            #date_factor_layer.append((bf_id , 1))
            #date_factor_layer.append((bf_id , 5))
            date_factor_layer.append((bf_id , layer))
        factor_layers.append( date_factor_layer )
        dates.append(now_date)

    factor_layer_df = pd.DataFrame(factor_layers, index = dates)
    factor_layer_df[pd.isnull(factor_layer_df)] = np.nan

    session.commit()
    session.close()

    factor_layer_df.to_csv('factor_layer_df.csv')
    return factor_layer_df


#根据各层秩相关选取因子
def regression_tree_ic_factor_layer_selector(bf_ids):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_layer_ic.trade_date, barra_stock_factor_layer_ic.bf_id, barra_stock_factor_layer_ic.ic).filter(barra_stock_factor_layer_ic.bf_id.in_(bf_ids)).statement
    layer_ic_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'bf_id'], parse_dates = ['trade_date'])
    layer_ic_df = layer_ic_df.unstack()
    layer_ic_df.columns = layer_ic_df.columns.droplevel(0)

    layer_ic_df = layer_ic_df.rolling(6).mean()
    layer_ic_abs = abs(layer_ic_df)

    #print layer_ic_df
    sql = session.query(barra_stock_factor_layer_stocks.layer, barra_stock_factor_layer_stocks.trade_date ,barra_stock_factor_layer_stocks.bf_id).statement
    all_factor_layer_stock_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'bf_id'])

    dates = []
    factor_layers = []
    for i in range(0, len(layer_ic_abs.index)):
        date = layer_ic_abs.index[i]
        ic = layer_ic_abs.loc[date]
        date_factor_layer = []
        for bf_id in ic.index:
            ic_v = ic.loc[bf_id]
            if ic_v >= 0.4:
                layers = all_factor_layer_stock_df.loc[date].loc[bf_id]
                max_layer = max(layers.layer)
                min_layer = min(layers.layer)
                if layer_ic_df.loc[date , bf_id] > 0:
                    layer = max_layer
                else:
                    layer = min_layer
                date_factor_layer.append((bf_id , layer))
            else:
                date_factor_layer.append(np.nan)
        factor_layers.append( date_factor_layer )
        dates.append(date)

    factor_layer_df = pd.DataFrame(factor_layers, index = dates)
    factor_layer_df[pd.isnull(factor_layer_df)] = np.nan

    #print factor_layer_df
    session.commit()
    session.close()

    return factor_layer_df



#根据秩相关选取因子，然后利用bootstrap计算净值
def factor_boot_pos():

    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012',
            'BF.000013','BF.000014','BF.000015','BF.000016','BF.000017']

    #bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012']
    factor_pos_df = regression_tree_ic_factor_layer_selector(bf_ids)


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


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()



    dates = factor_pos_df.index[7:]
    stock_pos_df = pd.DataFrame(0, index = dates, columns = all_stocks.globalid)

    for date in dates:
        factor_index_data = {}
        factor_stocks = {}
        date_factor_pos_df = factor_pos_df.loc[date].dropna()
        for bf_id in date_factor_pos_df:
            bf_id, layer = bf_id[0], bf_id[1]
            record = session.query(barra_stock_factor_layer_stocks.stock_ids).filter(and_(barra_stock_factor_layer_stocks.bf_id == bf_id, barra_stock_factor_layer_stocks.trade_date == date, barra_stock_factor_layer_stocks.layer == layer)).first()

            if record is None:
                continue

            stocks = json.loads(record[0])
            factor_stocks[(bf_id, layer)] = stocks

            pos = pd.DataFrame(1.0 / len(stocks), index = [date], columns = stocks)
            tmp_yield_df = yield_df[stocks]
            tmp_yield_df = tmp_yield_df[tmp_yield_df.index <= date]
            tmp_yield_df = tmp_yield_df.iloc[-120:,]
            pos = pos.reindex(tmp_yield_df.index).fillna(method = 'bfill')
            nav_df = (tmp_yield_df * pos).sum(axis = 1).dropna()
            nav_df = ( nav_df + 1 ).cumprod()
            factor_index_data[(bf_id,  layer)] = nav_df

        factor_index_df = pd.DataFrame(factor_index_data)
        df_inc = factor_index_df.pct_change().fillna(0.0)

        df_inc = df_inc.iloc[-120:,]

        bound = []
        for asset in df_inc.columns:
            bound.append({'sum1': 0,    'sum2' : 0,   'upper': 1.0,  'lower': 0.0})


        record = pd.Series(0, index = all_stocks.globalid)
        if len(df_inc.columns) == 0:
            record = 1.0 / len(record)
        else:
            if len(df_inc.columns) == 1:
                ws = [1.0]
            else:
                risk, returns, ws, sharpe = PF.markowitz_bootstrape(df_inc, bound, cpu_count=32, bootstrap_count=0)

            print date ,df_inc.columns, ws

            for i in range(0, len(ws)):
                #tmp_record = pd.Series(0, index = all_stocks.globalid)
                bf_id = df_inc.columns[i]
                stocks = factor_stocks[bf_id]
                stocks = list(set(stocks) & set(all_stocks.globalid.ravel()))
                w = ws[i]
                #sql = session.query(barra_stock_factor_exposure.stock_id, barra_stock_factor_exposure.factor_exposure).filter(and_(barra_stock_factor_exposure.bf_id == bf_id[0], barra_stock_factor_exposure.trade_date == date)).statement
                #factor_exposure_df = pd.read_sql(sql, session.bind, index_col = ['stock_id'])
                #factor_exposure_df = factor_exposure_df.loc[stocks]
                #factor_exposure_df = factor_exposure_df / factor_exposure_df.sum()
                #for stock_id in factor_exposure_df.index:
                #    record.loc[stock_id] = record.loc[stock_id] + w * factor_exposure_df.loc[stock_id].ravel()[0]
                for stock_id in stocks:
                    record.loc[stock_id] = record.loc[stock_id] + w * 1.0 / len(stocks)
        stock_pos_df.loc[date] = record

    session.commit()
    session.close()

    stock_pos_df.to_csv('stock_pos.csv')

    return stock_pos_df



def factor_pos_2_stock_pos(df):


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    stock_pos_df = None
    for bf_ids in df.columns:
        bf_id_vec = bf_ids.strip().split('.')
        layer = int(bf_id_vec[-1])
        bf_id = '.'.join(bf_id_vec[0:2])

        sql = session.query(barra_stock_factor_layer_stocks.trade_date, barra_stock_factor_layer_stocks.stock_ids).filter(and_(barra_stock_factor_layer_stocks.bf_id == bf_id, barra_stock_factor_layer_stocks.layer == layer)).statement

        layer_stock_df = pd.read_sql(sql, session.bind, index_col = ['trade_date'], parse_dates = ['trade_date'])

        stock_pos = {}
        ser = df[bf_ids]
        for date in ser.index:
            stock_pos[date] = json.loads(layer_stock_df.stock_ids.loc[date])

        bf_layer_stock_pos_df = stock_util.stock_pos_2_weight(stock_pos)
        for date in bf_layer_stock_pos_df.index:
            bf_layer_stock_pos_df.loc[date] = bf_layer_stock_pos_df.loc[date] * ser.loc[date]

        if stock_pos_df is None:
            stock_pos_df = bf_layer_stock_pos_df
        else:
            stock_pos_df = stock_pos_df + bf_layer_stock_pos_df


    session.commit()
    session.close()

    return stock_pos_df


#自由流通市值因子
def free_capital_factor():


    all_stocks = stock_util.all_stock_info()
    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_sharestruchg.compcode , tq_sk_sharestruchg.begindate , tq_sk_sharestruchg.enddate ,tq_sk_sharestruchg.fcircaamt).filter(tq_sk_sharestruchg.compcode.in_(all_stocks.sk_compcode)).statement


    #过滤掉未上市时的数据
    free_share_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['begindate', 'enddate']).dropna()
    free_share_df.enddate[free_share_df.enddate == '1900-01-01'] = datetime.now()
    free_share_df = free_share_df.rename(index = compcode_secode_dict)
    #三诺生物这个股票，同一时间段披露两个自由流通股数量，整个A股只有这一只股票如此，直接过滤掉，secode = 2010005089 
    free_share_df = free_share_df[free_share_df.index != '2010005089']

    sql = session.query(tq_qt_skdailyprice.secode, tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.tclose).filter(tq_qt_skdailyprice.secode.in_(all_stocks.index)).statement
    dailyprice_df = pd.read_sql(sql, session.bind, index_col = ['tradedate', 'secode'], parse_dates = ['tradedate']).replace(0.0, np.nan)

    dailyprice_df = dailyprice_df.unstack()
    dailyprice_df.columns = dailyprice_df.columns.droplevel(0)

    #dailyprice_df = pd.read_csv('barra_stock_factor/dailyprice.csv',index_col = ['tradedate'], parse_dates = ['tradedate']).replace(0.0, np.nan)


    free_capital_data = {}
    for secode, group in free_share_df.groupby(free_share_df.index):
        free_capitals = []
        for i in range(0 ,len(group)):
            record = group.iloc[i]
            begindate = record.begindate
            enddate = record.enddate
            fcircaamt = record.fcircaamt
            free_capital = dailyprice_df[secode][(dailyprice_df.index >= begindate) & (dailyprice_df.index <= enddate)] * fcircaamt
            free_capitals.append(free_capital)

        secode_free_capital = pd.concat(free_capitals, axis = 0)
        secode_free_capital = secode_free_capital.loc[dailyprice_df.index]
        free_capital_data[secode] = secode_free_capital


    free_capital_df = pd.DataFrame(free_capital_data).fillna(method = 'pad')
    free_capital_df = np.log(free_capital_df)

    #去极值标准化行业中性化
    free_capital_df = stock_factor_util.normalized(free_capital_df)
    #free_capital_df = industry_normalized(free_capital_df)

    #secode转成globalid
    free_capital_df = free_capital_df.rename(columns = secode_globalid_dict)
    free_capital_df = free_capital_df[free_capital_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    free_capital_df = stock_factor_util.valid_stock_filter(free_capital_df)
    stock_factor_util.update_factor_value('BF.000001', free_capital_df)
    #free_capital_df.to_csv('barra_stock_factor/free_capital_factor.csv')


    session.commit()
    session.close()

    return free_capital_df


#ep ttm因子
def ep_ttm_factor():

    all_stocks = stock_util.all_stock_info()
    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_finindic.tradedate ,tq_sk_finindic.secode, tq_sk_finindic.pettm).filter(tq_sk_finindic.secode.in_(all_stocks.index)).statement
    pe_ttm_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])
    pe_ttm_df = pe_ttm_df.unstack()
    pe_ttm_df.columns = pe_ttm_df.columns.droplevel(0)

    session.commit()
    session.close()

    ep_ttm_df = 1.0 / pe_ttm_df

    #去极值标准化行业中性化
    ep_ttm_df = stock_factor_util.normalized(ep_ttm_df)
    #ep_ttm_df = industry_normalized(ep_ttm_df)

    #secode转成globalid
    ep_ttm_df = ep_ttm_df.rename(columns = secode_globalid_dict)
    ep_ttm_df = ep_ttm_df[ep_ttm_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    ep_ttm_df = stock_factor_util.valid_stock_filter(ep_ttm_df)

    stock_factor_util.update_factor_value('BF.000002', ep_ttm_df)
    #ep_ttm_df.to_csv('barra_stock_factor/ep_ttm_factor.csv')

    return


#log市值因子
def ln_capital_factor():

    all_stocks = stock_util.all_stock_info()
    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_qt_skdailyprice.tradedate ,tq_qt_skdailyprice.secode, tq_qt_skdailyprice.totmktcap).filter(tq_qt_skdailyprice.secode.in_(all_stocks.index)).statement
    totmktcap_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])
    totmktcap_df = totmktcap_df.unstack()
    totmktcap_df.columns = totmktcap_df.columns.droplevel(0)
    totmktcap_df = np.log(totmktcap_df)

    session.commit()
    session.close()

    #去极值标准化行业中性化
    totmktcap_df = stock_factor_util.normalized(totmktcap_df)
    #ep_ttm_df = industry_normalized(ep_ttm_df)

    #secode转成globalid
    totmktcap_df = totmktcap_df.rename(columns = secode_globalid_dict)
    totmktcap_df = totmktcap_df[totmktcap_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    totmktcap_df = stock_factor_util.valid_stock_filter(totmktcap_df)
    totmktcap_df.to_csv('barra_stock_factor/totmktcap_factor.csv')


#bp因子
def bp_factor():

    all_stocks = stock_util.all_stock_info()

    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_finindic.tradedate ,tq_sk_finindic.secode, tq_sk_finindic.pb).filter(tq_sk_finindic.secode.in_(all_stocks.index)).statement
    pb_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])
    pb_df = pb_df.unstack()
    pb_df.columns = pb_df.columns.droplevel(0)

    bp_df = 1.0 / pb_df

    session.commit()
    session.close()

    #去极值标准化行业中性化
    bp_df = stock_factor_util.normalized(bp_df)
    #bp_df = industry_normalized(bp_df)

    #secode转成globalid
    bp_df = bp_df.rename(columns = secode_globalid_dict)
    bp_df = bp_df[bp_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    bp_df = stock_factor_util.valid_stock_filter(bp_df)

    stock_factor_util.update_factor_value('BF.000003', bp_df)
    #bp_df.to_csv('barra_stock_factor/bp_factor.csv')

    return bp_df


#对数收盘价因子
def ln_price_factor():

    all_stocks = stock_util.all_stock_info()

    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.secode  ,tq_qt_skdailyprice.tclose).filter(tq_qt_skdailyprice.secode.in_(all_stocks.index)).statement
    price_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])
    price_df = price_df.unstack()
    price_df.columns = price_df.columns.droplevel(0)
    price_df = np.log(price_df)

    session.commit()
    session.close()

    #去极值标准化行业中性化
    price_df = stock_factor_util.normalized(price_df)
    #bp_df = industry_normalized(bp_df)

    #secode转成globalid
    price_df = price_df.rename(columns = secode_globalid_dict)
    price_df = price_df[price_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    price_df = stock_factor_util.valid_stock_filter(price_df)

    stock_factor_util.update_factor_value('BF.000004', price_df)

    return price_df


#三月换手率因子
def turn_rate_factor():

    all_stocks = stock_util.all_stock_info()

    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.secode  ,tq_sk_yieldindic.turnrate3m).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement
    turn_rate_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])

    turn_rate_df = turn_rate_df.unstack()
    turn_rate_df.columns = turn_rate_df.columns.droplevel(0)

    #price_df = 1.0 / price_df

    turn_rate_df = turn_rate_df / 100.0

    session.commit()
    session.close()

    #去极值标准化行业中性化
    turn_rate_df = stock_factor_util.normalized(turn_rate_df)
    #bp_df = industry_normalized(bp_df)

    #secode转成globalid
    turn_rate_df = turn_rate_df.rename(columns = secode_globalid_dict)
    turn_rate_df = turn_rate_df[turn_rate_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    turn_rate_df = stock_factor_util.valid_stock_filter(turn_rate_df)

    stock_factor_util.update_factor_value('BF.000005', turn_rate_df)

    return turn_rate_df



#三个月成交金额因子
def trade_amount_factor():

    all_stocks = stock_util.all_stock_info()

    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_dquoteindic.tradedate, tq_sk_dquoteindic.secode  ,tq_sk_dquoteindic.amount).filter(tq_sk_dquoteindic.secode.in_(all_stocks.index)).statement
    trade_amount_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])

    trade_amount_df = trade_amount_df.unstack()
    trade_amount_df.columns = trade_amount_df.columns.droplevel(0)

    trade_amount_df = trade_amount_df.rolling(60).mean()
    trade_amount_df = trade_amount_df / 10000.0

    session.commit()
    session.close()

    #去极值标准化行业中性化
    trade_amount_df = stock_factor_util.normalized(trade_amount_df)
    #bp_df = industry_normalized(bp_df)

    #secode转成globalid
    trade_amount_df = trade_amount_df.rename(columns = secode_globalid_dict)
    trade_amount_df = trade_amount_df[trade_amount_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    trade_amount_df = stock_factor_util.valid_stock_filter(trade_amount_df)

    stock_factor_util.update_factor_value('BF.000006', trade_amount_df)

    return trade_amount_df


#三个月成交金额因子
def trade_amount_factor():

    all_stocks = stock_util.all_stock_info()

    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_dquoteindic.tradedate, tq_sk_dquoteindic.secode  ,tq_sk_dquoteindic.amount).filter(tq_sk_dquoteindic.secode.in_(all_stocks.index)).statement
    trade_amount_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])

    trade_amount_df = trade_amount_df.unstack()
    trade_amount_df.columns = trade_amount_df.columns.droplevel(0)

    trade_amount_df = trade_amount_df.rolling(60).mean()
    trade_amount_df = trade_amount_df / 10000.0

    session.commit()
    session.close()

    #去极值标准化行业中性化
    trade_amount_df = stock_factor_util.normalized(trade_amount_df)
    #bp_df = industry_normalized(bp_df)

    #secode转成globalid
    trade_amount_df = trade_amount_df.rename(columns = secode_globalid_dict)
    trade_amount_df = trade_amount_df[trade_amount_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    trade_amount_df = stock_factor_util.valid_stock_filter(trade_amount_df)

    stock_factor_util.update_factor_value('BF.000006', trade_amount_df)

    return trade_amount_df


#户均持股比例因子
def holder_avgpct_factor():

    all_stocks = stock_util.all_stock_info()

    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()


    sql = session.query(tq_sk_shareholdernum.enddate ,tq_sk_shareholdernum.publishdate, tq_sk_shareholdernum.compcode, tq_sk_shareholdernum.aholdproportionpacc).filter(tq_sk_shareholdernum.compcode.in_(all_stocks.sk_compcode)).statement
    holder_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['publishdate', 'enddate'])

    session.commit()
    session.close()

    holder_df = holder_df.rename(columns={'publishdate':'firstpublishdate'})
    holder_df = stock_factor_util.financial_report_data_trade_date(holder_df, 'aholdproportionpacc')

    #去极值标准化行业中性化
    holder_df = stock_factor_util.normalized(holder_df)
    #bp_df = industry_normalized(bp_df)

    holder_df = holder_df[holder_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    holder_df = stock_factor_util.valid_stock_filter(holder_df)

    stock_factor_util.update_factor_value('BF.000007', holder_df)

    return holder_df


#ROE ttm 因子
def roe_factor():

    all_stocks = stock_util.all_stock_info()

    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_prottmindic.enddate ,tq_fin_prottmindic.publishdate, tq_fin_prottmindic.compcode, tq_fin_prottmindic.roedilutedcut).filter(and_(tq_fin_prottmindic.reporttype == 3, tq_fin_prottmindic.compcode.in_(all_stocks.sk_compcode))).statement
    roe_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['publishdate', 'enddate'])

    session.commit()
    session.close()

    roe_df = roe_df.rename(columns={'publishdate':'firstpublishdate'})
    roe_df = stock_factor_util.financial_report_data_trade_date(roe_df, 'roedilutedcut')

    print roe_df
    #去极值标准化行业中性化
    roe_df = stock_factor_util.normalized(roe_df)
    #bp_df = industry_normalized(bp_df)

    roe_df = roe_df[roe_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    roe_df = stock_factor_util.valid_stock_filter(roe_df)

    stock_factor_util.update_factor_value('BF.000008', roe_df)

    return roe_df


#ROA ttm 因子
def roa_factor():

    all_stocks = stock_util.all_stock_info()

    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_prottmindic.enddate ,tq_fin_prottmindic.publishdate, tq_fin_prottmindic.compcode, tq_fin_prottmindic.roa).filter(and_(tq_fin_prottmindic.reporttype == 3, tq_fin_prottmindic.compcode.in_(all_stocks.sk_compcode))).statement
    roa_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['publishdate', 'enddate'])

    session.commit()
    session.close()

    roa_df = roa_df.rename(columns={'publishdate':'firstpublishdate'})
    roa_df = stock_factor_util.financial_report_data_trade_date(roa_df, 'roa')

    #去极值标准化行业中性化
    roa_df = stock_factor_util.normalized(roa_df)
    #bp_df = industry_normalized(bp_df)

    roa_df = roa_df[roa_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    roa_df = stock_factor_util.valid_stock_filter(roa_df)

    stock_factor_util.update_factor_value('BF.000009', roa_df)

    return roa_df


#三个月加权动量因子
def weight_strength_factor():


    all_stocks = stock_util.all_stock_info()

    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.secode ,tq_sk_yieldindic.turnrate, tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement

    weight_strength_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])

    weight_strength_df['weight_strength'] = weight_strength_df.turnrate * weight_strength_df.Yield
    weight_strength_df = weight_strength_df[['weight_strength']]

    weight_strength_df = weight_strength_df.unstack()
    weight_strength_df.columns = weight_strength_df.columns.droplevel(0)

    weight_strength_df = weight_strength_df.rolling(window = 60, min_periods = 30).mean()

    session.commit()
    session.close()

    #去极值标准化行业中性化
    weight_strength_df = stock_factor_util.normalized(weight_strength_df)
    #bp_df = industry_normalized(bp_df)

    #secode转成globalid
    weight_strength_df = weight_strength_df.rename(columns = secode_globalid_dict)
    weight_strength_df = weight_strength_df[weight_strength_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    weight_strength_df = stock_factor_util.valid_stock_filter(weight_strength_df)

    stock_factor_util.update_factor_value('BF.000010', weight_strength_df)

    return weight_strength_df


#三个月动量因子
def relative_strength_factor():

    all_stocks = stock_util.all_stock_info()

    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.secode , tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement

    relative_strength_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])

    relative_strength_df = relative_strength_df.unstack()
    relative_strength_df.columns = relative_strength_df.columns.droplevel(0)

    relative_strength_df = relative_strength_df.rolling(window = 60, min_periods = 30).mean()

    session.commit()
    session.close()

    #去极值标准化行业中性化
    relative_strength_df = stock_factor_util.normalized(relative_strength_df)
    #bp_df = industry_normalized(bp_df)

    #secode转成globalid
    relative_strength_df = relative_strength_df.rename(columns = secode_globalid_dict)
    relative_strength_df = relative_strength_df[relative_strength_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    relative_strength_df = stock_factor_util.valid_stock_filter(relative_strength_df)

    stock_factor_util.update_factor_value('BF.000011', relative_strength_df)

    return relative_strength_df


#三个月波动率因子
def std_factor():

    all_stocks = stock_util.all_stock_info()

    compcode_secode_dict = dict(zip(all_stocks.sk_compcode, all_stocks.index))
    secode_globalid_dict = dict(zip(all_stocks.index.ravel(), all_stocks.globalid.ravel()))

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_yieldindic.tradedate, tq_sk_yieldindic.secode , tq_sk_yieldindic.Yield).filter(tq_sk_yieldindic.secode.in_(all_stocks.index)).statement

    std_df = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])

    std_df = std_df.unstack()
    std_df.columns = std_df.columns.droplevel(0)

    std_df = std_df.rolling(window = 60, min_periods = 30).std()

    session.commit()
    session.close()

    #去极值标准化行业中性化
    std_df = stock_factor_util.normalized(std_df)
    #bp_df = industry_normalized(bp_df)

    #secode转成globalid
    std_df = std_df.rename(columns = secode_globalid_dict)
    std_df = std_df[std_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    std_df = stock_factor_util.valid_stock_filter(std_df)

    stock_factor_util.update_factor_value('BF.000012', std_df)

    return std_df



#流动比率因子
def current_ratio_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.currentrt).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    current_ratio_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    session.commit()
    session.close()

    current_ratio_df = stock_factor_util.financial_report_data_trade_date(current_ratio_df, 'currentrt')

    #去极值标准化行业中性化
    current_ratio_df = stock_factor_util.normalized(current_ratio_df)
    #bp_df = industry_normalized(bp_df)

    current_ratio_df = current_ratio_df[current_ratio_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    current_ratio_df = stock_factor_util.valid_stock_filter(current_ratio_df)

    stock_factor_util.update_factor_value('BF.000013', current_ratio_df)

    return current_ratio_df



#现金比率因子
def cash_ratio_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.cashrt).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    cash_ratio_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    session.commit()
    session.close()


    cash_ratio_df = stock_factor_util.financial_report_data_trade_date(cash_ratio_df, 'cashrt')

    #去极值标准化行业中性化
    cash_ratio_df = stock_factor_util.normalized(cash_ratio_df)
    #bp_df = industry_normalized(bp_df)

    cash_ratio_df = cash_ratio_df[cash_ratio_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    cash_ratio_df = stock_factor_util.valid_stock_filter(cash_ratio_df)

    stock_factor_util.update_factor_value('BF.000014', cash_ratio_df)

    return cash_ratio_df



#负债比率因子
def debtequityratio_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.ltmliabtota).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    debtequity_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    session.commit()
    session.close()

    debtequity_df = stock_factor_util.financial_report_data_trade_date(debtequity_df, 'ltmliabtota')

    #去极值标准化行业中性化
    debtequity_df = stock_factor_util.normalized(debtequity_df)
    #bp_df = industry_normalized(bp_df)

    debtequity_df = debtequity_df[debtequity_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    debtequity_df = stock_factor_util.valid_stock_filter(debtequity_df)

    stock_factor_util.update_factor_value('BF.000015', debtequity_df)

    return debtequity_df



#金融杠杆率因子
def finalcial_leverage_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.equtotliab).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    finalcial_leverage_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    finalcial_leverage_df.equtotliab = 1.0 / (1.0 + finalcial_leverage_df.equtotliab)

    session.commit()
    session.close()

    finalcial_leverage_df = stock_factor_util.financial_report_data_trade_date(finalcial_leverage_df, 'equtotliab')

    #去极值标准化行业中性化
    finalcial_leverage_df = stock_factor_util.normalized(finalcial_leverage_df)
    #bp_df = industry_normalized(bp_df)

    finalcial_leverage_df = finalcial_leverage_df[finalcial_leverage_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    finalcial_leverage_df = stock_factor_util.valid_stock_filter(finalcial_leverage_df)

    stock_factor_util.update_factor_value('BF.000016', finalcial_leverage_df)

    return finalcial_leverage_df



#毛利率因子
def grossprofit_factor():

    all_stocks = stock_util.all_stock_info()
    stock_listdate = stock_util.all_stock_listdate()[['sk_listdate']]
    all_stocks = pd.concat([all_stocks, stock_listdate], axis = 1, join_axes = [all_stocks.index])

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.enddate ,tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.compcode, tq_fin_proindicdata.reporttype ,tq_fin_proindicdata.sgpmargin).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode.in_(all_stocks.sk_compcode)).statement
    grossprofit_df = pd.read_sql(sql, session.bind, index_col = ['compcode'], parse_dates = ['firstpublishdate', 'enddate'])

    session.commit()
    session.close()
    #print grossprofit_df.iloc[:,0:15]

    grossprofit_df = stock_factor_util.financial_report_data_trade_date(grossprofit_df, 'sgpmargin')

    #去极值标准化行业中性化
    grossprofit_df = stock_factor_util.normalized(grossprofit_df)
    #bp_df = industry_normalized(bp_df)

    grossprofit_df = grossprofit_df[grossprofit_df.columns & all_stocks.globalid]

    #过滤掉不合法的因子值
    grossprofit_df = stock_factor_util.valid_stock_filter(grossprofit_df)

    stock_factor_util.update_factor_value('BF.000017', grossprofit_df)

    return grossprofit_df




#因子有效性检验
def stock_factor_availiability(bf_id):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_layer_stocks.trade_date, barra_stock_factor_layer_stocks.layer, barra_stock_factor_layer_stocks.stock_ids).filter(barra_stock_factor_layer_stocks.bf_id == bf_id).statement

    layer_stock_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'layer'], parse_dates = ['trade_date'])

    layer_stock_df = layer_stock_df.unstack()
    layer_stock_df.columns = layer_stock_df.columns.droplevel(0)

    session.commit()
    session.close()

    yield_df = pd.read_csv('barra_stock_factor/yield.csv', index_col = ['tradedate'], parse_dates = ['tradedate'])

    dates = list(set(yield_df.index & layer_stock_df.index))
    dates.sort()

    data = []
    for date in dates:
        layer_stocks = layer_stock_df.loc[date]
        layer_stds = []
        for layer in layer_stocks.index:
            stocks = json.loads(layer_stocks.loc[layer])
            yields = yield_df.loc[date, stocks]
            yields = yields.dropna()
            layer_stds.append(yields.std())

        data.append(layer_stds)

    df = pd.DataFrame(data, index = dates)
    df = df.rolling(60, min_periods = 30).mean()
    df.columns = [str(bf_id) + '_' + str(layer) for layer in layer_stocks.index]

    return df



