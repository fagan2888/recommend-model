#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

from ipdb import set_trace
import copy
import datetime
import json
import os
import sys
sys.path.append('shell')
import click
import pickle
import pandas as pd
import numpy as np
import logging
import logging.config
from numpy import mat
from pathos import multiprocessing
from scipy.stats import rankdata
from scipy.signal import hilbert
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial import distance
#from starvine.bvcopula.copula import frank_copula
import scipy
import scipy.optimize

from db import database, asset_barra_stock_factor_layer_nav, base_ra_index_nav, base_ra_index, base_trade_dates, asset_factor_cluster_nav, base_ra_fund, asset_factor_cluster
from db.asset_factor_cluster import *
from db import asset_factor_cluster, asset_fund, asset_stock, asset_stock_factor
# from sqlalchemy import MetaData, Table, select, func, literal_column
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
import DBData
import warnings
import factor_cluster, stock_util

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def setup_logging(
    default_path = './shell/logging.json',
    default_level = logging.INFO,
    env_key = 'LOG_CFG'):

    """Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)



@click.group(invoke_without_command=True)
@click.pass_context
def etf(ctx):
    '''ETF FUND
    '''


@etf.command()
@click.pass_context
def etf_fund(ctx):

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    #records = session.query(asset_fund.tq_fd_basicinfo.secode, asset_fund.tq_fd_basicinfo.fdname, asset_fund.tq_fd_basicinfo.fsymbol, asset_fund.tq_fd_basicinfo.fdtype).filter(asset_fund.tq_fd_basicinfo.fdnature == 'ETF').filter(asset_fund.tq_fd_basicinfo.fdstyle == 3).all()

    sql = session.query(asset_fund.tq_oa_securitymap.secode, asset_fund.tq_oa_securitymap.mapcode, asset_fund.tq_oa_securitymap.mapname).filter(asset_fund.tq_oa_securitymap.maptype == 25).filter(asset_fund.tq_oa_securitymap.enddate == '19000101').statement
    map_df = pd.read_sql(sql, session.bind, index_col = ['secode'])

    sql = session.query(asset_fund.tq_fd_basicinfo.secode, asset_fund.tq_fd_basicinfo.fdsname, asset_fund.tq_fd_basicinfo.fsymbol).filter(asset_fund.tq_fd_basicinfo.secode.in_(map_df.index)).statement
    fund_df = pd.read_sql(sql, session.bind, index_col = ['secode'])
    fund_index_df = pd.concat([fund_df, map_df], axis = 1, join_axes = [map_df.index])
    fund_index_df.reset_index(inplace = True)
    fund_index_df.set_index(['mapcode'], inplace = True)
    fund_index_df = fund_index_df.rename(columns={'secode':'fund_secode'})

    sql = session.query(asset_fund.tq_ix_basicinfo.secode, asset_fund.tq_ix_basicinfo.indexsname, asset_fund.tq_ix_basicinfo.symbol).filter(asset_fund.tq_ix_basicinfo.symbol.in_(fund_index_df.index)).statement
    index_info_df = pd.read_sql(sql, session.bind, index_col = ['symbol'])
    index_info_df = index_info_df.rename(columns={'secode':'index_secode'})
    fund_index_df = pd.concat([fund_index_df, index_info_df], axis = 1, join_axes = [fund_index_df.index])

    #print fund_index_df

    records = session.query(asset_fund.tq_fd_typeclass.securityid).filter(asset_fund.tq_fd_typeclass.l2codes == '200102').all()
    securityids = [record[0] for record in records]

    records = session.query(asset_fund.tq_oa_stcode.secode).filter(asset_fund.tq_oa_stcode.securityid.in_(securityids)).all()
    secodes = [record[0] for record in records]

    fund_index_df = fund_index_df[fund_index_df.fund_secode.isin(secodes)]

    available_index = set()
    for index_secode in set(fund_index_df.index_secode):
        record = session.query( asset_fund.tq_ix_mweight.tradedate ,asset_fund.tq_ix_mweight.constituentsecode, asset_fund.tq_ix_mweight.weight).filter(asset_fund.tq_ix_mweight.secode == index_secode).first()
        if record is None:
            continue
        else:
            available_index.add(index_secode)

    #print fund_index_df
    fund_index_df = fund_index_df[fund_index_df.index_secode.isin(available_index)]
    fund_index_df = fund_index_df.groupby(fund_index_df.index_secode).first()

    fund_index_df.index.name = 'index_secode'
    fund_index_df.to_csv('fund_index.csv')

    session.commit()
    session.close()



@etf.command()
@click.pass_context
def fund_stock_resolve(ctx):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()


    bf_ids = ['BF.000001.1','BF.000001.0','BF.000006.1', 'BF.000002.0', 'BF.000003.0', 'BF.000004.0', 'BF.000005.0','BF.000007.0','BF.000008.0','BF.000009.0','BF.000010.0','BF.000006.0','BF.000012.0', 'BF.000013.0', 'BF.000014.0', 'BF.000015.0', 'BF.000016.0', 'BF.000017.0', 'BF.000002.1', 'BF.000003.1', 'BF.000004.1', 'BF.000005.1','BF.000011.0','BF.000007.1','BF.000008.1','BF.000009.1','BF.000010.1','BF.000011.1','BF.000012.1', 'BF.000013.1', 'BF.000014.1', 'BF.000015.1', 'BF.000016.1', 'BF.000017.1']


    fund_index_df = pd.read_csv('fund_index.csv', index_col = ['index_secode'])
    index_secodes = fund_index_df.index

    bf_ids_stock = {}
    bf_ids_nav = {}
    for asset_id in bf_ids:

        asset_ids = asset_id.strip().split('.')
        layer = int(asset_ids[-1])
        bf_id = '.'.join(asset_ids[0:2])

        sql = session.query(asset_stock_factor.barra_stock_factor_layer_stocks.trade_date,  asset_stock_factor.barra_stock_factor_layer_stocks.stock_ids).filter(asset_stock_factor.barra_stock_factor_layer_stocks.bf_id == bf_id, asset_stock_factor.barra_stock_factor_layer_stocks.layer == layer).statement

        df = pd.read_sql(sql, session.bind, index_col = ['trade_date'], parse_dates = ['trade_date'])
        bf_ids_stock[asset_id] = df.stock_ids

        sql = session.query(asset_stock_factor.barra_stock_factor_layer_nav.trade_date,  asset_stock_factor.barra_stock_factor_layer_nav.nav).filter(asset_stock_factor.barra_stock_factor_layer_nav.bf_id == bf_id, asset_stock_factor.barra_stock_factor_layer_nav.layer == layer).statement
        df = pd.read_sql(sql, session.bind, index_col = ['trade_date'], parse_dates = ['trade_date'])
        bf_ids_nav[asset_id] = df.nav

    session.commit()
    session.close()

    bf_stock_df = pd.DataFrame(bf_ids_stock)
    bf_nav_df = pd.DataFrame(bf_ids_nav)
    bf_nav_df = bf_nav_df[bf_nav_df.index >= '2010-01-01']

    y_bf_id = bf_ids[0]
    #x_bf_ids = bf_ids[1:]

    dates = bf_stock_df[bf_stock_df.index >= '2015-01-01'].index

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    for trade_date in dates:

        all_stocks = stock_util.all_stock_info()
        y_stocks = json.loads(bf_stock_df.loc[trade_date, y_bf_id])
        record = pd.Series(0, index = all_stocks.globalid)
        record[record.index.isin(y_stocks)] = 1.0 / len(y_stocks)
        y_record = record.ravel()

        x_records = {}
        #for x_bf_id in x_bf_ids:
        #    x_stocks = json.loads(bf_stock_df.loc[trade_date, x_bf_id])
        #    record = pd.Series(0, index = all_stocks.globalid)
        #    record[record.index.isin(x_stocks)] = 1.0 / len(x_stocks)
        #    x_records[x_bf_id] = record

        for index_secode in index_secodes:
            indexsname = fund_index_df.loc[index_secode, 'indexsname']
            record = pd.Series(0, index = all_stocks.globalid)
            sql = session.query(asset_fund.tq_ix_mweight.constituentcode, asset_fund.tq_ix_mweight.weight).filter(and_(asset_fund.tq_ix_mweight.secode == index_secode, asset_fund.tq_ix_mweight.tradedate == trade_date.strftime('%Y%m%d'))).statement
            stock_code_weight_df = pd.read_sql(sql, session.bind, index_col = ['constituentcode'])
            if len(stock_code_weight_df) > 0:
                for code in stock_code_weight_df.index:
                    if 'SK.' + str(code) in record.index:
                        record['SK.' + str(code)] = stock_code_weight_df.loc[code, 'weight']
                record = record/ record.sum()
                x_records[indexsname] = record

        x_record_df = pd.DataFrame(x_records).fillna(0.0)

        if len(x_record_df) == 0:
            continue
        x_records = x_record_df.values
        x_records_num = len(x_record_df.columns)

        w = 1.0 * np.ones(x_records_num) / x_records_num
        bound = [ (0.0 , 1.0) for i in range(x_records_num)]
        constrain = ({'type':'eq', 'fun': lambda w: sum(w)-1.0 })
        result = scipy.optimize.minimize(fund_resolve_obj_func, w, (x_records, y_record), method='SLSQP', constraints=constrain, bounds=bound)
        ws = result.x

        print trade_date, (abs(y_record - np.dot(x_records, ws))).sum()
        for item in sorted(zip(ws, x_record_df.columns), reverse = True)[0:10]:
            print item[0], item[1]

        #tmp_bf_nav = bf_nav_df[bf_nav_df.index >= trade_date]
        #tmp_bf_inc = tmp_bf_nav.pct_change().fillna(0.0)
        #y = tmp_bf_inc[y_bf_id]
        #_y = pd.Series(np.dot(tmp_bf_inc[x_bf_ids], ws), index = tmp_bf_inc.index)
        #y_nav = (1 + y).cumprod()
        #_y_nav = (1 + _y).cumprod()

        #nav = pd.DataFrame(np.matrix([y_nav, _y_nav]).T, index = y_nav.index)
        #print y_nav.tail()
        #print _y_nav.tail()
        #print nav.tail()
        #nav.to_csv('nav.csv')
        #print trade_date, ws
        #print trade_date, (abs(y_record - np.dot(x_records, ws))).sum()
        #df = pd.DataFrame([y_record, np.dot(x_records, ws)], columns = all_stocks.globalid)
        #print df
        #df.T.to_csv('resolve.csv')

    session.commit()
    session.close()

def fund_resolve_obj_func(w, x_records, y_record):
    #print np.array(w).shape
    #print x_records.shape
    #print y_record.shape
    #print abs(y_record - np.dot(x_records, w)).sum()
    return ((y_record - np.dot(x_records, w)) ** 2).sum()
