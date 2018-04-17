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
from db import asset_stock_factor
from db import asset_stock
from db import asset_trade_dates
from multiprocessing import Pool
import math
import scipy.stats as stats
import json
from barra_stock_factor import *
from corr_regression_tree import *
import functools


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def bsf(ctx):
    '''multi factor
    '''
    pass


@bsf.command()
@click.pass_context
def barra_stock_factor_exposure(ctx):


    #ep_ttm_factor()
    #bp_factor()
    #free_capital_factor()
    #ln_price_factor()
    #turn_rate_factor()
    #trade_amount_factor()
    #holder_avgpct_factor()
    #roe_factor()
    #roa_factor()
    #weight_strength_factor()
    #relative_strength_factor()
    #std_factor()
    #current_ratio_factor()
    #cash_ratio_factor()
    #debtequityratio_factor()
    #finalcial_leverage_factor()
    #grossprofit_factor()
    #factor_funcs = [ep_ttm_factor, free_capital_factor, bp_factor]


    factor_funcs = [ep_ttm_factor, bp_factor, free_capital_factor, ln_price_factor, turn_rate_factor, trade_amount_factor, holder_avgpct_factor, roe_factor,
            roa_factor, weight_strength_factor, relative_strength_factor, std_factor, current_ratio_factor, cash_ratio_factor, debtequityratio_factor, 
            finalcial_leverage_factor, grossprofit_factor]
    pool = Pool(len(factor_funcs))
    for func in factor_funcs:
        pool.apply_async(func)
    pool.close()
    pool.join()


    #ln_price_factor()
    return



@bsf.command()
@click.pass_context
def barra_stock_factor_yield(ctx):

    free_capital_factor_df = pd.read_csv('barra_stock_factor/free_capital_factor.csv', index_col =['tradedate'], parse_dates = ['tradedate'])
    ep_factor_df = pd.read_csv('barra_stock_factor/ep_ttm_factor.csv', index_col =['tradedate'], parse_dates = ['tradedate'])
    factor_yield(free_capital_factor_df, ep_factor_df)

    return


@bsf.command()
@click.pass_context
def barra_stock_factor_layer(ctx):


    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012',
            'BF.000013', 'BF.000014', 'BF.000015', 'BF.000016', 'BF.000017']

    pool = Pool(20)
    #pool.map(factor_layer_stocks, bf_ids)
    pool.map(regression_tree_factor_layer, bf_ids)
    pool.close()
    pool.join()

    #regression_tree_factor_layer(bf_ids[0])

    return


@bsf.command()
@click.pass_context
def barra_stock_factor_corr_regression_tree_layer(ctx):


    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012',
            'BF.000013', 'BF.000014', 'BF.000015', 'BF.000016', 'BF.000017']

    regression_tree_factor_corr_layer(bf_ids)

    return


@bsf.command()
@click.pass_context
def barra_stock_factor_fund(ctx):


    bf_ids = ['BF.000001.0', 'BF.000002.0', 'BF.000003.0', 'BF.000004.0', 'BF.000005.0','BF.000006.0','BF.000007.0','BF.000008.0','BF.000009.0','BF.000010.0','BF.000011.0','BF.000012.0', 'BF.000013.0', 'BF.000014.0', 'BF.000015.0', 'BF.000016.0', 'BF.000017.0', 'BF.000001.1', 'BF.000002.1', 'BF.000003.1', 'BF.000004.1', 'BF.000005.1','BF.000006.1','BF.000007.1','BF.000008.1','BF.000009.1','BF.000010.1','BF.000011.1','BF.000012.1', 'BF.000013.1', 'BF.000014.1', 'BF.000015.1', 'BF.000016.1', 'BF.000017.1']

    pool = Pool(40)
    # pool.map(layer_index_fund, bf_ids)
    layer_index_fund(bf_ids[0])
    pool.close()
    pool.join()

    #layer_index_fund(bf_ids[0])

    return


@bsf.command()
@click.pass_context
def barra_stock_factor_layer_index_nav(ctx):


    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012',
            'BF.000013','BF.000014','BF.000015','BF.000016','BF.000017']

    # pool = Pool(20)
    # pool.map(factor_regression_tree_layer_nav, bf_ids)
    # pool.close()
    # pool.join()

    factor_regression_tree_layer_nav(bf_ids[0])

    return


@bsf.command()
@click.pass_context
def barra_stock_factor_layer_index_weight_nav(ctx):


    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012']

    #pool = Pool(16)
    #pool.map(factor_layer_weight_nav, bf_ids)
    #pool.close()
    #pool.join()


    factor_layer_weight_nav('BF.000001')

    return


@bsf.command()
@click.pass_context
def barra_stock_factor_corr_selector(ctx):

    #bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012']

    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012',
            'BF.000013','BF.000014','BF.000015','BF.000016','BF.000017']
    #corr_factor_layer_selector(bf_ids)
    regression_tree_ic_factor_layer_selector(bf_ids)
    return


@bsf.command()
@click.pass_context
def barra_stock_factor_ic(ctx):

    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012',
            'BF.000013','BF.000014','BF.000015','BF.000016','BF.000017']

    pool = Pool(20)
    pool.map(regression_tree_factor_layer_ic, bf_ids)
    pool.close()
    pool.join()

    # regression_tree_factor_layer_ic(bf_ids[3])
    return


@bsf.command()
@click.pass_context
def barra_stock_factor_fund_ic(ctx):

    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012',
            'BF.000013','BF.000014','BF.000015','BF.000016','BF.000017']

    # pool = Pool(20)
    # pool.map(regression_tree_factor_layer_fund_ic, bf_ids)
    # pool.close()
    # pool.join()

    regression_tree_factor_layer_fund_ic(bf_ids[0])
    return


@bsf.command()
@click.pass_context
def barra_factor_boot_pos(ctx):

    factor_boot_pos()

    return




@bsf.command()
@click.pass_context
def barra_stock_valid(ctx):

    stock_factor_util.stock_valid_table()

    return


@bsf.command()
@click.pass_context
def barra_stock_factor_cluster(ctx):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003']

    sql = session.query(asset_stock_factor.barra_stock_factor_exposure.bf_id, asset_stock_factor.barra_stock_factor_exposure.stock_id, asset_stock_factor.barra_stock_factor_exposure.trade_date, asset_stock_factor.barra_stock_factor_exposure.factor_exposure).filter(asset_stock_factor.barra_stock_factor_exposure.bf_id.in_(bf_ids)).statement
    factor_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'stock_id', 'bf_id'], parse_dates = ['trade_date'])

    #factor_df = pd.read_csv('./barra_stock_factor/bf_stock_factor.csv', index_col = ['trade_date', 'stock_id'], parse_dates = ['trade_date'])

    session.commit()
    session.close()

    multi_factor_kmeans(factor_df)

    return


@bsf.command()
@click.pass_context
def barra_stock_factor_yield_allocate(ctx):

    ln_capital_yield = pd.read_csv('totmktcap_r.csv', index_col = ['tradedate'], parse_dates = ['tradedate'])
    ep_ttm_yield = pd.read_csv('ep_r.csv', index_col = ['tradedate'], parse_dates = ['tradedate'])

    factor_yield = pd.concat([ln_capital_yield, ep_ttm_yield], axis = 1)
    factor_yield = factor_yield.dropna()

    factor_yield = factor_yield[factor_yield.index >= '2014-01-01']
    factor_nav = ( 1 + factor_yield ).cumprod()

    #print factor_nav
    factor_nav.to_csv('factor_nav.csv')


@bsf.command()
@click.pass_context
def barra_stock_factor_availiability(ctx):

    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003']
    dfs = []
    for bf_id in bf_ids:
        df = stock_factor_availiability(bf_id)
        dfs.append(df)

    #stock_factor_availiability(bf_ids[1])
    df = pd.concat(dfs, axis = 1)
    df.to_csv('layer_std.csv')
    return


@bsf.command()
@click.pass_context
def barra_regression_tree_spliter(ctx):
    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012',
            'BF.000013','BF.000014','BF.000015','BF.000016','BF.000017']
    #regression_tree_factor_spliter(bf_ids)
    #regression_tree_factor_layer('BF.000001')
    return




@bsf.command()
@click.pass_context
def export_barra_stock_factor_nav(ctx):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()


    sql = session.query(barra_stock_factor_layer_nav.trade_date, barra_stock_factor_layer_nav.bf_id ,barra_stock_factor_layer_nav.layer, barra_stock_factor_layer_nav.nav).statement

    df = pd.read_sql(sql ,session.bind, index_col = ['trade_date', 'bf_id','layer'], parse_dates = ['trade_date'])

    session.commit()
    session.close()

    df = df.unstack([1,2])
    df.columns = df.columns.droplevel(0)
    df = df.dropna()
    df = df.resample('W-FRI').last()
    df_inc = df.pct_change().fillna(0.0)


    #df_inc = df_inc.loc[:,['BF.000001', 'BF.000006']]
    #df_inc = df_inc.iloc[:,[1,3]]
    df_inc = df_inc.loc[:,'BF.000001']
    df_inc = df_inc.iloc[:,1]

    #print np.corrcoef(df_inc.iloc[:,0], df_inc.iloc[:,1])
    #print np.corrcoef(pow( ( 1 + df_inc.iloc[:,0]), len(df_inc)), pow( ( 1 + df_inc.iloc[:,1]), len(df_inc)))


    session.commit()
    session.close()

    zz500 = base_ra_index_nav.load_series(120000002)
    zz500 = zz500.loc[df_inc.index]
    zz500_inc = zz500.pct_change().fillna(0.0)

    print np.corrcoef(df_inc, zz500_inc)
    print np.corrcoef(np.exp(df_inc * 52), np.exp(zz500_inc * 52))

    juchao_xiao = base_ra_index_nav.load_series(120000020)
    juchao_xiao = juchao_xiao.loc[df_inc.index]
    juchao_xiao_inc =   juchao_xiao.pct_change().fillna(0.0)

    #print np.corrcoef(juchao_xiao_inc, zz500_inc)
    #print np.corrcoef(juchao_xiao_inc - juchao_xiao_inc.mean(), zz500_inc - zz500_inc.mean())
    #print np.corrcoef(pow( ( 1 + juchao_xiao_inc), 52), pow( ( 1 + zz500_inc), 52))
    #print np.corrcoef(np.exp(juchao_xiao_inc * 52), np.exp(zz500_inc * 52))

    #print np.corrcoef(juchao_xiao_inc, zz500_inc)
    return



@bsf.command()
@click.pass_context
def export_barra_stock_factor_ic(ctx):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_layer_ic.trade_date, barra_stock_factor_layer_ic.bf_id, barra_stock_factor_layer_ic.ic).statement

    df = pd.read_sql(sql ,session.bind, index_col = ['trade_date', 'bf_id'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    df = df.unstack()
    df.columns = df.columns.droplevel(0)

    df = abs(df)
    df = df.rolling(12).mean()
    df.to_csv('ic.csv')

    df = df.dropna()
    df = df.mean(axis = 0)
    print df
    return


@bsf.command()
@click.pass_context
def barra_stock_factor_turnover(ctx):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_valid_factor.trade_date, barra_stock_factor_valid_factor.bf_layer_id).statement

    df = pd.read_sql(sql ,session.bind, index_col = ['trade_date', 'bf_layer_id'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    df['tag'] = 1
    df = df.unstack()
    df.columns = df.columns.droplevel(0)
    df = df[df.index >= '2012-07-01']

    #print df.sum(axis = 1)

    #print df.apply(np.nansum, axis = 1)
    #print df.div(df.apply(np.nansum, axis = 1), axis = 0)
    df = df.div(df.apply(np.nansum, axis = 1), axis = 0).fillna(0.0)
    print df.diff().abs().fillna(0.0).sum().sum() / 2



@bsf.command()
@click.pass_context
def fund_pool_turnover(ctx):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_valid_factor.trade_date, barra_stock_factor_valid_factor.bf_layer_id).statement

    df = pd.read_sql(sql ,session.bind, index_col = ['trade_date', 'bf_layer_id'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    df['tag'] = 1
    df = df.unstack()
    df.columns = df.columns.droplevel(0)
    df = df[df.index >= '2012-07-01']


    #print df.apply(np.nansum, axis = 1)
    print df.div(df.apply(np.nansum, axis = 1), axis = 0)
    df = df.div(df.apply(np.nansum, axis = 1), axis = 0).fillna(0.0)
    print df.diff().abs().fillna(0.0).sum().sum() / 2





@bsf.command()
@click.pass_context
def barra_stock_factor_2_composite_asset(ctx):


    bf_ids = ['BF.000001.0', 'BF.000002.0', 'BF.000003.0', 'BF.000004.0', 'BF.000005.0','BF.000006.0','BF.000007.0','BF.000008.0','BF.000009.0','BF.000010.0','BF.000011.0','BF.000012.0', 'BF.000013.0', 'BF.000014.0', 'BF.000015.0', 'BF.000016.0', 'BF.000017.0', 'BF.000001.1', 'BF.000002.1', 'BF.000003.1', 'BF.000004.1', 'BF.000005.1','BF.000006.1','BF.000007.1','BF.000008.1','BF.000009.1','BF.000010.1','BF.000011.1','BF.000012.1', 'BF.000013.1', 'BF.000014.1', 'BF.000015.1', 'BF.000016.1', 'BF.000017.1']

    pass



@bsf.command()
@click.pass_context
def valid_factor_performance(ctx):

    bf_ids = ['BF.000001', 'BF.000002', 'BF.000003', 'BF.000004', 'BF.000005','BF.000006','BF.000007','BF.000008','BF.000009','BF.000010','BF.000011','BF.000012',
            'BF.000013','BF.000014','BF.000015','BF.000016','BF.000017']


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_layer_ic.trade_date, barra_stock_factor_layer_ic.bf_id, barra_stock_factor_layer_ic.ic).filter(barra_stock_factor_layer_ic.bf_id.in_(bf_ids)).statement
    layer_ic_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'bf_id'], parse_dates = ['trade_date'])
    layer_ic_df = layer_ic_df.unstack()
    layer_ic_df.columns = layer_ic_df.columns.droplevel(0)


    layer_ic_df = layer_ic_df.rolling(14).mean()
    layer_ic_abs = abs(layer_ic_df)

    sql = session.query(barra_stock_factor_layer_stocks.layer, barra_stock_factor_layer_stocks.trade_date ,barra_stock_factor_layer_stocks.bf_id).statement
    all_factor_layer_stock_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'bf_id'])

    dates = []
    factor_layers = []
    for i in range(0, len(layer_ic_abs.index)):
        date = layer_ic_abs.index[i]
        ic = layer_ic_abs.loc[date]
        ic = ic.sort_values(ascending = False)
        ic = ic.dropna()
        if len(ic) <= 4:
            continue
        threshold = ic.iloc[4]
        date_factor_layer = []
        for bf_id in ic.index:
            ic_v = ic.loc[bf_id]
            if ic_v >= threshold:
                if layer_ic_df.loc[date , bf_id] > 0:
                    date_factor_layer.append(bf_id + '.1')
                else:
                    date_factor_layer.append(bf_id + '.0')

        factor_layers.append( date_factor_layer )
        dates.append(date)

    factor_layer_df = pd.DataFrame(factor_layers, index = dates)
    factor_layer_df[pd.isnull(factor_layer_df)] = np.nan
    factor_layer_df = factor_layer_df[factor_layer_df.index >= '2010-01-01']

    layer_nav_dfs = []
    for bf_id in bf_ids:
        sql = session.query(barra_stock_factor_layer_nav.trade_date, barra_stock_factor_layer_nav.layer, barra_stock_factor_layer_nav.nav).filter(barra_stock_factor_layer_nav.bf_id == bf_id).statement
        layer_nav_df = pd.read_sql(sql, session.bind, index_col = ['trade_date', 'layer'], parse_dates = ['trade_date'])

        layer_nav_df = layer_nav_df.unstack()
        layer_nav_df.columns = layer_nav_df.columns.droplevel(0)
        cols = []
        for col in layer_nav_df.columns:
            cols.append(bf_id + '.' + str(col))

        layer_nav_df.columns = cols
        #print layer_nav_df.tail()
        layer_nav_dfs.append(layer_nav_df)

    layer_nav_df = pd.concat(layer_nav_dfs, axis = 1)
    layer_nav_df = layer_nav_df[layer_nav_df.index >= '2010-01-01']

    dates = factor_layer_df.index
    all_ranks = []
    interval = 1
    for i in range(len(dates) - interval):
        start_date = dates[i]
        end_date = dates[i + interval]

        tmp_layer_nav_df = layer_nav_df[layer_nav_df.index <= end_date]
        tmp_layer_nav_df = tmp_layer_nav_df[tmp_layer_nav_df.index >= start_date]
        tmp_layer_nav_df = tmp_layer_nav_df.iloc[-1] / tmp_layer_nav_df.iloc[0] - 1

        tmp_layer_nav_df = tmp_layer_nav_df.sort_values(ascending = False)

        valid_factors = factor_layer_df.loc[start_date]

        factor_sorted = list(tmp_layer_nav_df.index.ravel())
        ranks = []
        for factor in valid_factors.values:
            ranks.append(factor_sorted.index(factor))
        print start_date, np.mean(ranks)
        all_ranks.append(np.mean(ranks))

    print np.mean(all_ranks)

    session.commit()
    session.close()


