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
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund, asset_factor_cluster, asset_stock_factor
from db.asset_stock_factor import *
from db.asset_stock import *
from db import asset_trade_dates, asset_ra_pool, asset_factor_cluster
from multiprocessing import Pool
import math
import scipy.stats as stats
import json
import statsmodels.api as sm
import statsmodels
import Portfolio as PF
import DBData
import Financial as fin
import Const


from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.tree import DecisionTreeRegressor


import stock_util
import stock_factor_util
import CommandPool



logger = logging.getLogger(__name__)



def get_adjust_point(startdate = '2012-01-01', enddate=None, label_period=13):

    # 加载时间轴数据
    if not enddate:
        yesterday = (datetime.now() - timedelta(days=1));
        enddate = yesterday.strftime("%Y-%m-%d")


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()


    sql = session.query(asset_factor_cluster.factor_cluster_nav.factor_selected_date).filter(and_(asset_factor_cluster.factor_cluster_nav.factor_selected_date >= startdate, asset_factor_cluster.factor_cluster_nav.factor_selected_date <= enddate)).statement
    df = pd.read_sql(sql, session.bind, index_col = ['factor_selected_date'], parse_dates = ['factor_selected_date'])
    df = df.groupby(df.index).first()
    df = df.sort_index()

    #records = session.query(trade_dates.td_date).filter(and_(trade_dates.td_date >= startdate, trade_dates.td_date <= enddate)).all()
    #print records

    session.commit()
    session.close()

    return df



#根据指数和基金列表筛选基金
def factor_cluster_fund_pool(factor_cluster_ids_corr, codes, lookback, limit):


    factor_cluster_ids = list(factor_cluster_ids_corr.keys())

    for cluster_id in factor_cluster_ids:
        delete_cluster_fund(cluster_id)


    pre_cluster_funds = {}

    for day in get_adjust_point().index:

        dates = base_trade_dates.trade_date_lookback_index(end_date=day, lookback=lookback)

        df_nav_fund = base_ra_fund_nav.load_daily(dates[0], day, codes = codes)
        df_nav_fund = df_nav_fund.loc[dates]
        df_nav_fund = df_nav_fund.dropna(axis = 1, thresh = len(df_nav_fund) * 2.0 / 3)

        df_nav_index = {}
        for cluster_id in factor_cluster_ids:
            df_nav_index[cluster_id] = asset_factor_cluster.load_selected_factor_series(cluster_id, reindex=dates, begin_date=dates[0], end_date=day)


        df_nav_index = pd.DataFrame(df_nav_index)

        df_nav_index = df_nav_index.dropna(axis = 1, how = 'all')


        df_nav_index_fund = pd.concat([df_nav_index, df_nav_fund], axis = 1, join_axes = [df_nav_index.index])

        df_inc_index_fund = df_nav_index_fund.pct_change().fillna(0.0)

        corr = df_inc_index_fund.corr()

        fund_best_cluster = {}

        for fund_code in set(df_inc_index_fund.columns).difference(set(factor_cluster_ids)):

            best_fund_cluster_corr = 0.0
            best_cluster_id = None

            for cluster_id in df_nav_index.columns:
                fund_cluster_corr = corr.loc[cluster_id, fund_code]
                #print day, cluster_id, fund_code, fund_cluster_corr

                if fund_cluster_corr > best_fund_cluster_corr:
                    best_cluster_id = cluster_id
                    best_fund_cluster_corr = fund_cluster_corr

            funds = fund_best_cluster.setdefault(best_cluster_id, {})

            funds[fund_code] = best_fund_cluster_corr


        for cluster_id in fund_best_cluster.keys():

            if cluster_id is None:
                continue

            fund_corrs = fund_best_cluster[cluster_id]

            fund_corrs_ser = pd.Series(fund_corrs)

            fund_corrs_ser.sort_values(inplace=True, ascending=False)

            #print day, cluster_id, fund_corrs_ser.head(10)

            fund_corrs_ser = fund_corrs_ser[fund_corrs_ser >= factor_cluster_ids_corr[cluster_id]]

            fund_jensens = {}
            for fund_code in fund_corrs_ser.index:
                fund_jensens[fund_code] = fin.jensen(df_inc_index_fund[fund_code], df_inc_index_fund[cluster_id] ,Const.rf)

            fund_jensens_ser = pd.Series(fund_jensens)

            fund_jensens_ser.sort_values(inplace=True, ascending=False)

            pre_funds = pre_cluster_funds.setdefault(cluster_id, [])

            if len(pre_funds) == 0:

                print day, cluster_id, fund_jensens_ser.head(limit).index.ravel()

                #save_cluster_fund(day, cluster_id, fund_jensens_ser.head(limit).index.ravel())

                pre_funds = fund_jensens_ser.head(limit).index.ravel()

            else:

                fund_jensens_candidate = fund_jensens_ser.iloc[0 : len(fund_jensens_ser) / 4]

                fund_candidate = fund_jensens_candidate.index.ravel()

                fund_codes = []

                for code in pre_funds:
                    if code in fund_candidate:
                        fund_codes.append(code)

                for code in fund_candidate:
                    if code not in pre_funds:
                        fund_codes.append(code)

                print day, cluster_id, fund_codes[0:limit]

                #save_cluster_fund(day, cluster_id, fund_codes[0: limit])

                pre_funds = fund_codes[0: limit]

            pre_cluster_funds[cluster_id] = pre_funds

    return 0



#根据选择的有效因子计算基金池
def barra_stock_factor_fund_pool(lookback, limit):


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_valid_factor.trade_date, barra_stock_factor_valid_factor.bf_layer_id).statement

    valid_factor_df = pd.read_sql(sql, session.bind, index_col = ['trade_date'], parse_dates = ['trade_date'])

    session.commit()
    session.close()

    adjust_points = valid_factor_df.groupby(valid_factor_df.index).first().index

    adjust_points = adjust_points[adjust_points >= '2012-01-01']

    pre_fund_pool = {}

    #adjust_points = adjust_points[adjust_points >= '2014-01-01']
    for day in adjust_points:

        valid_factors = valid_factor_df.loc[day, 'bf_layer_id'].ravel()
        codes = base_ra_fund.find_type_fund(1).ra_code.ravel()


        dates = base_trade_dates.trade_date_lookback_index(end_date=day, lookback=lookback)
        df_nav_fund = base_ra_fund_nav.load_daily(dates[0], day, codes = codes)
        df_nav_fund = df_nav_fund.loc[dates]
        df_nav_fund = df_nav_fund.dropna(axis = 1, thresh = len(df_nav_fund) * 2.0 / 3)


        df_nav_index = {}
        for bf_id in valid_factors:
            df_nav_index[bf_id] = load_factor_nav_series(bf_id, reindex = dates, begin_date = dates[0], end_date = day)

        df_nav_index = pd.DataFrame(df_nav_index)

        df_nav_index_fund = pd.concat([df_nav_index, df_nav_fund], axis = 1, join_axes = [df_nav_index.index])

        df_inc_index_fund = df_nav_index_fund.pct_change().fillna(0.0)

        #计算相关性
        corr = df_inc_index_fund.corr()


        #计算每个因子可使用的基金
        bf_avaible_funds = {}
        for bf_id in valid_factors:
            bf_corr = corr.loc[bf_id]
            bf_corr = bf_corr.loc[bf_corr.index.difference(valid_factors)]
            bf_corr.sort_values(inplace=True, ascending=False)


            if len(bf_corr[bf_corr >= 0.85]) < 10:
                bf_corr = bf_corr.head(12)
            else:
                bf_corr = bf_corr[bf_corr >= 0.85]
            bf_corr.sort_values(inplace = True, ascending = False)

            fund_jensens = {}
            for fund_code in bf_corr.index:
                fund_jensens[fund_code] = fin.jensen(df_inc_index_fund[fund_code], df_inc_index_fund[bf_id] ,Const.rf)
            fund_jensens_ser = pd.Series(fund_jensens)
            fund_jensens_ser.sort_values(inplace=True, ascending=False)
            bf_avaible_funds[bf_id] = fund_jensens_ser.iloc[0 : len(fund_jensens_ser) / 2].index.ravel()

            #print bf_id, bf_corr.loc[bf_avaible_funds[bf_id]]

        this_fund_pool = {}
        used_codes = []

        for bf_id in valid_factors:

            pre_funds = pre_fund_pool.setdefault(bf_id, [])

            if len(pre_funds) > 0:

                this_funds = this_fund_pool.setdefault(bf_id, [])
                avaible_funds = bf_avaible_funds[bf_id]
                for code in pre_funds:
                    if code in avaible_funds:
                        this_funds.append(code)
                        used_codes.append(code)


        for k, v in pre_fund_pool.items():
            for code in v:
                for bf_id in valid_factors:
                    avaibel_funds = bf_avaible_funds[bf_id]
                    if code in avaibel_funds:
                        this_funds = this_fund_pool.setdefault(bf_id, [])
                        if (len(this_funds) < limit) and (code not in used_codes):
                            this_funds.append(code)
                            used_codes.append(code)

        for bf_id in valid_factors:

            this_funds = this_fund_pool.setdefault(bf_id, [])
            avaible_funds = bf_avaible_funds[bf_id]
            for code in avaible_funds:
                if (len(this_funds) < limit) and (code not in used_codes):
                    this_funds.append(code)
                    used_codes.append(code)


        pre_fund_pool = this_fund_pool

        print day, pd.Series(this_fund_pool)

        save_factor_cluster_fund_pool(day, 'FC.000002.3', pd.Series(this_fund_pool))

    return 0




#根据选择的有效因子计算基金池
def barra_stock_factor_fund_pool_duplicated(lookback, limit):


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_valid_factor.trade_date, barra_stock_factor_valid_factor.bf_layer_id).statement

    valid_factor_df = pd.read_sql(sql, session.bind, index_col = ['trade_date'], parse_dates = ['trade_date'])

    session.commit()
    session.close()

    adjust_points = valid_factor_df.groupby(valid_factor_df.index).first().index

    adjust_points = adjust_points[adjust_points >= '2012-01-01']


    #adjust_points = adjust_points[adjust_points >= '2014-01-01']
    pre_fund_pool = {}
    for day in adjust_points:

        valid_factors = valid_factor_df.loc[day, 'bf_layer_id'].ravel()
        codes = base_ra_fund.find_type_fund(1).ra_code.ravel()


        dates = base_trade_dates.trade_date_lookback_index(end_date=day, lookback=lookback)
        df_nav_fund = base_ra_fund_nav.load_daily(dates[0], day, codes = codes)
        df_nav_fund = df_nav_fund.loc[dates]
        df_nav_fund = df_nav_fund.dropna(axis = 1, thresh = len(df_nav_fund) * 2.0 / 3)


        df_nav_index = {}
        for bf_id in valid_factors:
            df_nav_index[bf_id] = load_selected_factor_nav_series(bf_id, reindex = dates, begin_date = dates[0], end_date = day)

        df_nav_index = pd.DataFrame(df_nav_index)

        df_nav_index_fund = pd.concat([df_nav_index, df_nav_fund], axis = 1, join_axes = [df_nav_index.index])

        df_inc_index_fund = df_nav_index_fund.pct_change().fillna(0.0)

        #计算相关性
        corr = df_inc_index_fund.corr()


        #计算每个因子可使用的基金
        bf_avaible_funds = {}
        for bf_id in valid_factors:
            bf_corr = corr.loc[bf_id]
            bf_corr = bf_corr.loc[bf_corr.index.difference(valid_factors)]
            bf_corr.sort_values(inplace=True, ascending=False)


            bf_corr = bf_corr[bf_corr < 1.0]
            bf_corr = bf_corr[bf_corr >= 0.85]
            bf_corr.sort_values(inplace = True, ascending = False)

            fund_jensens = {}
            for fund_code in bf_corr.index:
                fund_jensens[fund_code] = fin.jensen(df_inc_index_fund[fund_code], df_inc_index_fund[bf_id] ,Const.rf)
            fund_jensens_ser = pd.Series(fund_jensens)
            fund_jensens_ser.sort_values(inplace=True, ascending=False)
            num = len(fund_jensens_ser) / 5 if len(fund_jensens_ser) / 5 >= 10 else 10
            bf_avaible_funds[bf_id] = fund_jensens_ser.iloc[0 : num].index.ravel()

        this_fund_pool = {}

        for bf_id in valid_factors:

            pre_funds = pre_fund_pool.setdefault(bf_id, [])

            if len(pre_funds) > 0:

                this_funds = this_fund_pool.setdefault(bf_id, [])
                avaible_funds = bf_avaible_funds[bf_id]
                for code in pre_funds:
                    if code in avaible_funds:
                        this_funds.append(code)


        #print this_fund_pool
        for bf_id in valid_factors:

            bf_corr = corr.loc[bf_id]
            bf_corr = bf_corr.loc[bf_corr.index.difference(valid_factors)]
            bf_corr.sort_values(inplace=True, ascending=False)


            #if len(bf_corr[bf_corr >= 0.85]) < 10:
            #    bf_corr = bf_corr.head(12)
            #else:
            #    bf_corr = bf_corr[bf_corr >= 0.85]
            bf_corr = bf_corr[bf_corr < 1.0]
            bf_corr = bf_corr[bf_corr >= 0.85]
            bf_corr.sort_values(inplace = True, ascending = False)

            fund_jensens = {}
            for fund_code in bf_corr.index:
                fund_jensens[fund_code] = fin.jensen(df_inc_index_fund[fund_code], df_inc_index_fund[bf_id] ,Const.rf)
            fund_jensens_ser = pd.Series(fund_jensens)
            fund_jensens_ser.sort_values(inplace=True, ascending=False)

            #print bf_id, bf_corr.loc[bf_avaible_funds[bf_id]]

            this_funds = this_fund_pool.setdefault(bf_id, [])
            for code in fund_jensens_ser.index:
                if code not in this_funds:
                    this_funds.append(code)

            #for code in bf_corr.index[0:5]:
            #    this_funds.append(code)

        for bf_id in this_fund_pool.keys():
            this_fund_pool[bf_id] = this_fund_pool[bf_id][0:5]


        pre_fund_pool = this_fund_pool
        print day, pd.Series(this_fund_pool)


        save_factor_cluster_fund_pool(day, 'FC.000002.3', pd.Series(this_fund_pool))

    return 0



def delete_cluster_fund(day, cluster_id):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    record = session.query(asset_ra_pool.ra_pool.id).filter(asset_ra_pool.ra_pool.ra_index_id == cluster_id).first()

    if record is None:
        return

    pool_id = record[0]

    session.query(asset_ra_pool.ra_pool_fund).filter(asset_ra_pool.ra_pool_fund.ra_pool == pool_id).filter(asset_ra_pool.ra_pool_fund.ra_date == day).delete()

    session.commit()
    session.close()



def save_cluster_fund(day, cluster_id, fund_codes):

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(base_ra_fund.ra_fund.ra_code, base_ra_fund.ra_fund.globalid).filter(base_ra_fund.ra_fund.ra_code.in_(fund_codes)).statement
    code_id_df = pd.read_sql(sql, session.bind, index_col = ['ra_code'])
    session.commit()
    session.close()


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    record = session.query(asset_ra_pool.ra_pool.id).filter(asset_ra_pool.ra_pool.ra_index_id == cluster_id).first()
    pool_id = record[0]

    for code in code_id_df.index:

        fund_id = code_id_df.loc[code, 'globalid']
        rpf = asset_ra_pool.ra_pool_fund()
        rpf.ra_pool = pool_id
        rpf.ra_category = 0
        rpf.ra_fund_id = fund_id
        rpf.ra_fund_code = code
        rpf.ra_date = day

        session.merge(rpf)

    session.commit()
    session.close()




def save_factor_cluster_fund_pool(day, factor_cluster_id, stock_factor_funds):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    records = session.query(asset_factor_cluster.factor_cluster_struct.fc_subject_asset_id).filter(asset_factor_cluster.factor_cluster_struct.fc_parent_cluster_id == factor_cluster_id).all()


    cluster_ids = [record[0] for record in records]

    for cluster_id in cluster_ids:
        delete_cluster_fund(day, cluster_id)


    pool_funds = {}

    for stock_factor_id in stock_factor_funds.index:

        #print stock_factor_id
        record = session.query(asset_factor_cluster.factor_cluster_struct.fc_parent_cluster_id).filter(asset_factor_cluster.factor_cluster_struct.fc_subject_asset_id == stock_factor_id).filter(asset_factor_cluster.factor_cluster_struct.fc_parent_cluster_id.in_(cluster_ids)).first()
        if record is None:
            continue
        cluster_id = record[0]

        record = session.query(asset_ra_pool.ra_pool.id).filter(asset_ra_pool.ra_pool.ra_index_id == cluster_id).first()
        pool_id = record[0]

        fund_codes = stock_factor_funds.loc[stock_factor_id]

        funds = pool_funds.setdefault(pool_id, set())

        for code in fund_codes:
            funds.add(code)

    for pool_id in pool_funds.keys():
        fund_codes = pool_funds[pool_id]

        base_engine = database.connection('base')
        base_Session = sessionmaker(bind=base_engine)
        base_session = base_Session()
        sql = base_session.query(base_ra_fund.ra_fund.ra_code, base_ra_fund.ra_fund.globalid).filter(base_ra_fund.ra_fund.ra_code.in_(fund_codes)).statement
        code_id_df = pd.read_sql(sql, base_session.bind, index_col = ['ra_code'])
        base_session.commit()
        base_session.close()


        for code in code_id_df.index:


            fund_id = code_id_df.loc[code, 'globalid']
            rpf = asset_ra_pool.ra_pool_fund()
            rpf.ra_pool = pool_id
            rpf.ra_category = 0
            rpf.ra_fund_id = fund_id
            rpf.ra_fund_code = code
            rpf.ra_date = day

            session.add(rpf)

        session.commit()

    session.commit()
    session.close()
