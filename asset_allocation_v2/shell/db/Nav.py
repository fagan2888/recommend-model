#coding=utf8

import logging
import pandas as pd
import numpy as np
import datetime
import calendar
# import pdb
from sqlalchemy import *

from itertools import groupby
from operator import itemgetter

from db import database
from util.xdebug import dd

logger = logging.getLogger(__name__)

class Nav(object):

    def __init__(self):
        self.tabs = {}

    def select(self, xtype, gids, sdate=None, edate=None):
        (db, s) = (None, None)
        if xtype == 3:
            #
            # 基金资产
            #
            db = database.connection('base')
            t = self.tabs.setdefault(xtype, Table('ra_fund_nav', MetaData(bind=db), autoload=True))

            columns = [
                t.c.ra_fund_id.label('ra_asset_id'),
                t.c.ra_date,
                t.c.ra_nav_adjusted.label('ra_nav'),
            ]

            s = select(columns).where(t.c.ra_fund_id.in_(gids))
            if sdate is not None:
                s = s.where(t.c.ra_date >= sdate)
            if edate is not None:
                s = s.where(t.c.ra_date <= edate)


        elif xtype == 4:
            #
            # 修型资产
            #
            db = database.connection('asset')
            t = self.tabs.setdefault(xtype, Table('rs_reshape_nav', MetaData(bind=db), autoload=True))

            columns = [
                t.c.rs_reshape_id.label('ra_asset_id'),
                t.c.rs_date.label('ra_date'),
                t.c.rs_nav.label('ra_nav'),
            ]

            s = select(columns).where(t.c.rs_reshape_id.in_(gids))

            if sdate is not None:
                s = s.where(t.c.rs_date >= sdate)
            if edate is not None:
                s = s.where(t.c.rs_date <= edate)

        elif xtype == 12:
            #
            # 指数资产
            #
            db = database.connection('base')
            t = Table('ra_index_nav', MetaData(bind=db), autoload=True)

            columns = [
                t.c.ra_index_id.label('ra_asset_id'),
                t.c.ra_date,
                t.c.ra_nav,
            ]

            s = select(columns).where(t.c.ra_index_id.in_(gids))

            if sdate is not None:
                s = s.where(t.c.ra_date >= sdate)
            if edate is not None:
                s = s.where(t.c.ra_date <= edate)
        else:
            pass


        return (db, s)

    def load_pool_nav(self, gids, sdate, edate):
        db = database.connection('asset')
        t = self.tabs.setdefault(1, Table('ra_pool_nav', MetaData(bind=db), autoload=True))

        for asset_id in gids:
            #
            # 基金池资产
            #
            asset_id %= 10000000
            (pool_id, category) = (asset_id // 100, asset_id % 100)
            ttype = pool_id // 10000

            columns = [
                t.c.ra_date,
                t.c.ra_nav,
            ]

            s = select(columns) \
                .where(t.c.ra_pool == id_) \
                .where(t.c.ra_category == category) \
                .where(t.c.ra_type == xtype)

            if sdate is not None:
                s = s.where(t.c.ra_date >= sdate)
            if edate is not None:
                s = s.where(t.c.ra_date <= edate)

            df = pd.read_sql(s, db, index_col = ['ra_date'], parse_dates=['ra_date'])

            data[asset_id] = df['ra_nav']

        return pd.DataFrame(data)


    def load(self, gids, reindex=None, sdate=None, edate=None):

        dfs = []
        for xtype, v in groupby(gids, key = lambda x: x // 10000000):
            if (xtype == 1):
                df = self.load_pool_nav(gids, sdate=sdate, edate=edate)
            else:
                (db, s) = self.select(xtype, v, sdate, edate)
                if s is not None:
                    df = pd.read_sql(s, db, index_col=['ra_asset_id', 'ra_date'], parse_dates=['ra_date'])
                    df = df.unstack(0)
                    df.columns = df.columns.droplevel(0)
            if df is not None:
                dfs.append(df)

        df_result = pd.concat(dfs, axis=1)
        if  reindex is not None:
            df_result = df_result.reindex(reindex, method='pad')

        return df_result

    def load_tdate_and_nav(self, gids, sdate=None, edate=None):
        result = {}
        index = pd.DatetimeIndex(pd.date_range(sdate, edate))
        db = database.connection('base')
        tdate = Table('trade_dates', MetaData(bind=db), autoload=True)
        for xtype, v in groupby(gids, key = lambda x: x // 10000000):
            if xtype == 3:
                #
                # 基金资产
                #
                t = self.tabs.setdefault(xtype, Table('ra_fund_nav', MetaData(bind=db), autoload=True))
                columns = [
                    t.c.ra_fund_id.label('ra_asset_id'),
                    t.c.ra_nav,
                    t.c.ra_nav_adjusted,
                    t.c.ra_type,
                    t.c.ra_date,
                ]

                # s = select(columns).where(t.c.ra_fund_id.in_(gids)).where(t.c.ra_mask.op('&')(0x01) == 0)
                s = select(columns) \
                    .select_from(t.join(tdate, t.c.ra_date == tdate.c.td_date)) \
                    .where(t.c.ra_fund_id.in_(gids))

                if sdate is not None:
                    s = s.where(t.c.ra_date >= sdate)
                if edate is not None:
                    s = s.where(t.c.ra_date <= edate)

                df = pd.read_sql(s, db, index_col=['ra_asset_id', 'ra_date'], parse_dates=['ra_date'])
                # if 33027398 in gids:
                #     dd(df.loc[33027398])

                # pdb.set_trace()
                if not df.loc[df['ra_type'] == 3, 'ra_nav'].empty:
                    df.loc[df['ra_type'] == 3, 'ra_nav'] = df['ra_nav_adjusted']
                df.drop(['ra_type', 'ra_nav_adjusted'], axis=1, inplace=True)

                for asset_id in df.index.levels[0]:
                    # print "load nav", asset_id
                    df_nav = df.loc[asset_id].copy()
                    df_nav['ra_nav_date'] = df_nav.index
                    df_nav = df_nav.reindex(index, method='bfill').fillna(method='ffill')

                    result[asset_id] = df_nav

        return result

    def load_nav_and_date(self, gids, sdate=None, edate=None):
        dfs = []
        for xtype, v in groupby(gids, key = lambda x: x // 10000000):
            if xtype == 3:
                #
                # 基金资产
                #
                db = database.connection('base')
                t = self.tabs.setdefault(xtype, Table('ra_fund_nav', MetaData(bind=db), autoload=True))

                columns = [
                    t.c.ra_fund_id.label('ra_asset_id'),
                    t.c.ra_nav,
                    t.c.ra_nav_adjusted,
                    t.c.ra_type,
                    t.c.ra_date,
                ]

                s = select(columns).where(t.c.ra_fund_id.in_(gids)).where(t.c.ra_mask.op('&')(0x01) == 0)
                if sdate is not None:
                    s = s.where(t.c.ra_date >= sdate)
                if edate is not None:
                    s = s.where(t.c.ra_date <= edate)

                df = pd.read_sql(s, db, index_col=['ra_asset_id', 'ra_date'], parse_dates=['ra_date'])

                # pdb.set_trace()
                if not df.loc[df['ra_type'] == 3, 'ra_nav'].empty:
                    df.loc[df['ra_type'] == 3, 'ra_nav'] = df['ra_nav_adjusted']
                df.drop(['ra_type', 'ra_nav_adjusted'], axis=1, inplace=True)

                dfs.append(df)

        if len(dfs) == 1:
            result = dfs[0]
        else:
            result = pd.concat(dfs)

        return result

