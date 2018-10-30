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

    return df, df_nav


def load_SP_feature():

    now = datetime.now()
    start = '2000-01-01'

    # Economy and Monetary Indicator
    df = pd.read_excel('data/sp_macro_data.xls', index_col=0, parse_dates=True)
    df.columns = ['PE', 'FFTR', 'VIX', 'CPI', 'CCPI', 'GDP', 'UE', 'YS', 'CCI', 'ICI', 'EOI']
    df = df.resample('m').last()
    df = df.fillna(method='pad')

    # Derivative Indicator
    df['DVIX'] = df['VIX'].diff(3)
    df['RO20'] = df['PE'] + df['CPI']

    df = df.shift(1)
    df = df[df.index < now]
    df = df[df.index > start]

    # Market Indicator
    YDs = 365
    df_nav = pd.read_excel('data/sp500.xls', index_col=0, parse_dates=True)['nav']
    df_ret_1y = df_nav.pct_change(YDs).resample('m').last()
    df_ret_1y = df_ret_1y.reindex(df.index).to_frame('R1y')
    df_ret_2y = df_nav.pct_change(2*YDs).resample('m').last()
    df_ret_2y = df_ret_2y.reindex(df.index).to_frame('R2y')

    # Concat
    df = pd.concat([df, df_ret_1y, df_ret_2y], 1)

    # Adjust Direction
    df['PE'] = -df['PE']
    df['FFTR'] = -df['FFTR']
    df['VIX'] = -df['VIX']
    df['CPI'] = -df['CPI']
    df['CCPI'] = -df['CCPI']
    df['R1y'] = -df['R1y']
    df['R2y'] = -df['R2y']
    df['DVIX'] = -df['DVIX']
    df['RO20'] = -df['RO20']

    df['GDP'] = -df['GDP']
    df['UE'] = -df['UE']

    return df, df_nav


def load_HK_feature():

    now = datetime.now()

    # Economy and Monetary Indicator
    df_A_feature, _ = load_A_feature()
    df_SP_feature, _ = load_SP_feature()

    df_A_feature = df_A_feature[['GDP', 'EGDP', 'CPI', 'ECPI', 'UEGDP', 'UECPI']]
    df_SP_feature = df_SP_feature[['FFTR', 'CPI', 'CCPI']]
    df = pd.merge(df_A_feature, df_SP_feature, left_index=True, right_index=True, how='inner')

    df = df.shift(1)
    df = df[df.index <= now]

    # Market Indicator
    YDs = 365
    df_nav = base_ra_index_nav.load_series('120000015')
    df_ret_1y = df_nav.pct_change(YDs).reindex(df.index).to_frame('R1y')
    df_ret_2y = df_nav.pct_change(2*YDs).reindex(df.index).to_frame('R2y')

    # Concat
    df = pd.concat([df, df_ret_1y, df_ret_2y], 1)
    df['R1y'] = -df['R1y']
    df['R2y'] = -df['R2y']

    # Adjust Direction

    return df, df_nav



