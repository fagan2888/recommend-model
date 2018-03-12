#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
#coding=utf8

from sqlalchemy import MetaData, Table, select, func, distinct
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database
import MySQLdb
import config

from dateutil.parser import parse
from ipdb import set_trace

logger = logging.getLogger(__name__)

#
# base.ra_index_nav
#
def load_series(id_, layer, reindex=None, begin_date=None, end_date=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('barra_stock_factor_layer_nav', metadata, autoload=True)

    columns = [
        t1.c.trade_date.label('date'),
        t1.c.nav.label('nav'),
    ]

    s = select(columns).where(t1.c.bf_id == id_).where(t1.c.layer == layer)

    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)

    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['nav']


def load_series_2(id_, reindex=None, begin_date=None, end_date=None):
    layer = int(id_[-1])
    id_ = id_[:-2]
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('barra_stock_factor_layer_nav', metadata, autoload=True)

    columns = [
        t1.c.trade_date.label('date'),
        t1.c.nav.label('nav'),
    ]

    s = select(columns).where(t1.c.bf_id == id_).where(t1.c.layer == layer)

    if begin_date is not None:
        s = s.where(t1.c.trade_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.trade_date <= end_date)

    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['nav']


def load_layer_id():
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('barra_stock_factor_layer_nav', metadata, autoload=True)

    columns = [
        t1.c.bf_id,
        t1.c.layer,
    ]

    s = select([distinct(t1.c.bf_id), t1.c.layer])

    df = pd.read_sql(s, db)
    #df = df.

    return df


def load_all_nav():
    pool = []
    bf_layer = load_layer_id()
    for idx, row in bf_layer.iterrows():
        pool.append('{}.{}'.format(row['bf_id'], row['layer']))

    assets = []
    asset_names = []
    for asset in pool:
        id_ = asset[:-2]
        layer = int(asset[-1])
        tmp_df = load_series(id_, layer)
        assets.append(tmp_df)
        asset_names.append(asset)

    df = pd.concat(assets, 1)
    df.columns = asset_names
    df = df.dropna()

    return df


if __name__ == '__main__':
    bf_nav = load_all_nav()
    bf_nav.to_csv('barra_nav.csv', index_label = 'date')
