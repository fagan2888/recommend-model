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
from ipdb import set_trace

import config
from db import database, asset_trade_dates, base_ra_index_nav
from db.asset_fundamental import *
from calendar import monthrange
from datetime import datetime, timedelta
from scipy.stats import entropy
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from ipdb import set_trace
logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def rv(ctx):
    '''
    macro timing
    '''
    if ctx.invoked_subcommand is None:
        ctx.invoke(feature_thresh_update)
        # ctx.invoke(rela_view_update)
    else:
        pass


@rv.command()
@click.pass_context
def rela_view_update(ctx):

    pass


@rv.command()
@click.pass_context
def feature_thresh_update(ctx):

    df_feature = load_A_feature()
    forcast_horizon = 3
    df_nav = base_ra_index_nav.load_series('120000016')
    df_label = cal_label(df_nav, forcast_horizon)
    ser_feature_thresh = cal_feature_thresh(df_feature, df_nav, df_label)
    df_view = cal_view(df_feature, ser_feature_thresh)
    test_feature(df_feature, df_label, ser_feature_thresh)


def load_A_feature():

    now = datetime.now()

    # Economy and Monetary Indicator
    df = pd.read_excel('data/ashare_macro_data.xls', index_col=0, parse_dates=True)
    df.columns = ['GDP', 'EGDP', 'CPI', 'ECPI', 'SR', 'HLR', 'RR']
    df = df.resample('m').last()
    df = df.fillna(method='pad')
    df = df.shift(1)
    df = df[df.index < now]

    # Market Indicator
    YDs = 365
    df_nav = base_ra_index_nav.load_series('120000016')
    df_ret_1y = df_nav.pct_change(YDs).reindex(df.index).to_frame('R1y')
    df_ret_2y = df_nav.pct_change(2*YDs).reindex(df.index).to_frame('R2y')

    # Concat
    df = pd.concat([df, df_ret_1y, df_ret_2y], 1)

    # Adjust Direction
    df['UEGDP'] = df['GDP'] - df['EGDP']
    df['UECPI'] = -(df['CPI'] - df['ECPI'])
    df['SR'] = -df['SR']
    df['HLR'] = -df['HLR']
    df['RR'] = -df['RR']
    df['R1y'] = -df['R1y']
    df['R2y'] = -df['R2y']

    return df


def cal_label(df_nav, forcast_horizon):

    df_ret = df_nav.pct_change()
    df_ret_month = df_ret.resample('m').sum()
    df_ret_month = df_ret_month.rolling(forcast_horizon).sum()
    df_ret_month = df_ret_month.shift(-forcast_horizon)
    df_ret_month = df_ret_month.dropna()

    return df_ret_month


def cal_feature_thresh(df_feature, df_nav, df_label):

    df_ret_month = df_label

    features = ['UEGDP', 'UECPI', 'SR', 'HLR', 'RR', 'R1y', 'R2y']
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
        print(feature, fea_thresh, wr, len(dfr1))

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
        ave = ((dfr1+0.1)**2).sum() + ((dfr2-0.1)**2).sum()
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
    df_view = df_view.sum(1) / df_view.count(1)
    df_view = 1 - 2*df_view


def test_feature(df_feature, df_label, ser_feature_thresh):

    df_label = df_label.to_frame('y')
    indicators = ser_feature_thresh.index
    for indicator in indicators:
        df_anova = df_feature[indicator]
        df_anova = df_anova.dropna()
        t = ser_feature_thresh.loc[indicator]
        arr_anova = np.where(df_anova < t, 1, 0)
        df_anova = pd.DataFrame(data=arr_anova, index=df_anova.index, columns=['Variety'])
        df_anova = pd.merge(df_anova, df_label, left_index=True, right_index=True, how='inner')
        model = ols('y~Variety', df_anova).fit()
        anovat = anova_lm(model)
        print()
        print('##########################################################################################')
        print(indicator)
        print(anovat)
        print('##########################################################################################')

    set_trace()



