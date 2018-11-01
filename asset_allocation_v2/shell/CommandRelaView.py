#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import sys
import click
sys.path.append('shell')
import logging
import pandas as pd
import numpy as np
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from calendar import monthrange
from datetime import datetime, timedelta
from scipy.stats import entropy
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData, Table, select
from ipdb import set_trace

import config
from db import database, asset_trade_dates, base_ra_index_nav
from db.asset_fundamental import *
from RelaViewHelper import load_A_feature, load_SP_feature, load_HK_feature


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def rv(ctx):
    '''
    macro timing
    '''
    if ctx.invoked_subcommand is None:
        ctx.invoke(feature_thresh_update, market='CN')
        ctx.invoke(feature_thresh_update, market='US')
        ctx.invoke(feature_thresh_update, market='HK')
        # ctx.invoke(rela_view_update)
    else:
        pass


@rv.command()
@click.pass_context
def rela_view_update(ctx):

    pass


@rv.command()
@click.pass_context
@click.option('--market', 'market', default='CN', help='China/US/HK')
@click.option('--start-date', 'start_date', default='2000-01-01', help='View start date')
@click.option('--viewid', 'viewid', default='BL.000003', help='BL View id')
def feature_thresh_update(ctx, market, start_date, viewid):

    # market = 'China'
    # market = 'US'
    # market = 'HK'
    print()
    print(market)

    market_assets = {
        'CN': ['120000001', '120000002', '120000016', '120000053', '120000056', '120000058', '120000073', 'MZ.FA0010', 'MZ.FA0050', 'MZ.FA0070', 'MZ.FA1010', 'ALayer'],
        'US': ['120000013', '120000020'],
        'HK': ['120000015']
    }

    if market == 'CN':
        df_feature, df_nav = load_A_feature()
    elif market == 'US':
        df_feature, df_nav = load_SP_feature()
    elif market == 'HK':
        df_feature, df_nav = load_HK_feature()

    forcast_horizon = 12
    df_label = cal_label(df_nav, forcast_horizon)
    ser_feature_thresh = cal_feature_thresh(df_feature, df_nav, df_label)
    ser_feature_thresh = test_feature(df_feature, df_label, ser_feature_thresh)
    df_view = cal_view(df_feature, ser_feature_thresh)
    df_view = df_view[df_view.index >= start_date]
    df_view = df_view * len(ser_feature_thresh)

    df_view = df_view.reset_index()
    df_view.columns = ['bl_date', 'bl_view']
    df_view['globalid'] = 'BL.000003'
    df_view = df_view[['globalid', 'bl_date', 'bl_view']]
    df_view['created_at'] = datetime.now()
    df_view['updated_at'] = datetime.now()
    view_assets = market_assets[market]
    for index_id in view_assets:

        df_new = df_view
        df_new['bl_index_id'] = index_id
        df_new = df_new.set_index(['globalid', 'bl_date', 'bl_index_id'])

        db = database.connection('asset')
        metadata = MetaData(bind=db)
        t = Table('ra_bl_view', metadata, autoload=True)
        columns = [
            t.c.globalid,
            t.c.bl_date,
            t.c.bl_view,
            t.c.bl_index_id,
            t.c.created_at,
            t.c.updated_at,
        ]
        s = select(columns).where(t.c.globalid == viewid).where(t.c.bl_index_id == index_id)
        df_old = pd.read_sql(s, db, index_col=['globalid', 'bl_date', 'bl_index_id'], parse_dates=['bl_date'])
        database.batch(db, t, df_new, df_old, timestamp=False)

        # print(df_new.tail())


def cal_label(df_nav, forcast_horizon):

    df_ret = df_nav.pct_change()
    df_ret_month = df_ret.resample('m').sum()
    df_ret_month = df_ret_month.rolling(forcast_horizon).sum()
    df_ret_month = df_ret_month.shift(-forcast_horizon)
    df_ret_month = df_ret_month.dropna()

    return df_ret_month


def cal_feature_thresh(df_feature, df_nav, df_label):

    df_ret_month = df_label

    features = df_feature.columns
    # features = ['UEGDP', 'UECPI', 'SR', 'HLR', 'RR', 'R1y', 'R2y']
    # features = ['HLR', 'RR']
    feature_thresh = {}
    for feature in features:
        df_fea = df_feature[feature].copy()
        df_fea = df_fea.dropna()
        dfr = df_ret_month.copy()
        common_dates = df_fea.index.intersection(dfr.index)
        df_fea = df_fea.loc[common_dates]
        dfr = dfr.loc[common_dates]
        fea_thresh = cal_thresh(df_fea, dfr)

        dfr1 = dfr[df_fea < fea_thresh]
        wr = len(dfr1[dfr1 < 0]) / len(dfr1)
        # print(feature, fea_thresh, wr, len(dfr1))

        if wr > 0.5:
            feature_thresh[feature] = fea_thresh
    ser_feature_thresh = pd.Series(feature_thresh)

    return ser_feature_thresh


def cal_thresh(df_fea, dfr):

    unique_feas = np.unique(df_fea)
    min_ave = np.inf
    threshold = 0
    for t in unique_feas[1:]:
        dfr1 = dfr[df_fea < t]
        dfr2 = dfr[df_fea >= t]
        ave = ((dfr1+1)**2).sum() + ((dfr2-1)**2).sum()
        if ave < min_ave:
            min_ave = ave
            threshold = t

    return threshold


def cal_thresh2(df_fea, dfr):

    unique_feas = np.unique(df_fea)
    min_ave = np.inf
    threshold = 0
    for t in unique_feas[1:]:
        dfr1 = dfr[df_fea < t]
        # dfr2 = dfr[df_fea >= t]
        ave = len(dfr1[dfr1 > 0]) / len(dfr1)
        if ave < min_ave:
            min_ave = ave
            threshold = t

    return threshold


def cal_view(df_feature, ser_feature_thresh):

    indicators = ser_feature_thresh.index
    df_feature = df_feature[indicators]
    df_view = {}
    for indicator in indicators:
        ser_indicator = df_feature[indicator]
        ser_indicator = ser_indicator.dropna()
        t = ser_feature_thresh.loc[indicator]
        arr_indicator = np.where(ser_indicator < t, 1, 0)
        ser_indicator = pd.Series(arr_indicator, index=ser_indicator.index)
        df_view[indicator] = ser_indicator

    df_view = pd.DataFrame(df_view)
    print(df_view.tail())
    df_view = df_view.sum(1) / df_view.count(1)
    df_view = 1 - 2*df_view

    return df_view


def test_feature(df_feature, df_label, ser_feature_thresh):

    df_label = df_label.to_frame('y')
    indicators = ser_feature_thresh.index
    ser_feature_thresh_new = {}
    for indicator in indicators:
        df_anova = df_feature[indicator]
        df_anova = df_anova.dropna()
        t = ser_feature_thresh.loc[indicator]
        arr_anova = np.where(df_anova < t, 1, 0)
        df_anova = pd.DataFrame(data=arr_anova, index=df_anova.index, columns=['Variety'])
        df_anova = pd.merge(df_anova, df_label, left_index=True, right_index=True, how='inner')
        model = ols('y~Variety', df_anova).fit()
        anovat = anova_lm(model)
        FP = anovat.loc['Variety', 'PR(>F)']
        if FP < 0.1:
            ser_feature_thresh_new[indicator] = t
            print(indicator, t)

            # print()
            # print('##########################################################################################')
            # print(indicator)
            # print(anovat)
            # print('##########################################################################################')

    ser_feature_thresh_new = pd.Series(ser_feature_thresh_new)
    return ser_feature_thresh_new


