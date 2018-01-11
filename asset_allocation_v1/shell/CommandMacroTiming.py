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
from db import database, asset_trade_dates
from db.asset_fundamental import *
from calendar import monthrange
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def mt(ctx):
    '''
    macro timing
    '''
    pass


@mt.command()
@click.pass_context
def macro_view(ctx):
    rev = re_view()
    irv = ir_view()
    epsv = eps_view()


def re_view():

    return 0


def ir_view():
    ytm = load_10Y_bond_ytm()

    return 0


def eps_view():
#    eps_mean = load_eps_mean()
#    ngdp = load_ngdp_yoy()

    return 0


def load_m1_yoy():

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(
        t_macro_msupply.nyear,
        t_macro_msupply.nmonth,
        t_macro_msupply.growthrate_m1).statement
    m1_yoy = pd.read_sql(sql, session.bind)

    session.commit()
    session.close()

    dates = []
    for y, m in zip(m1_yoy.nyear.values, m1_yoy.nmonth.values):
        d = monthrange(y, m)[1]
        date = datetime(y, m, d)
        dates.append(date)

    m1_yoy.index = dates
    # 由于宏观数据滞后公布，因此最新数据为NAN
    m1_yoy = m1_yoy.dropna()
    m1_yoy = m1_yoy.sort_index()
#    m1_yoy.to_csv('data/m1.csv', index_label = 'date')

    return m1_yoy


def load_m2_yoy():

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(
        t_macro_msupply.nyear,
        t_macro_msupply.nmonth,
        t_macro_msupply.growthrate_m2).statement
    m2_yoy = pd.read_sql(sql, session.bind)

    session.commit()
    session.close()

    dates = []
    for y, m in zip(m2_yoy.nyear.values, m2_yoy.nmonth.values):
        d = monthrange(y, m)[1]
        date = datetime(y, m, d)
        dates.append(date)

    m2_yoy.index = dates
    m2_yoy = m2_yoy.sort_index()
#    m2_yoy.to_csv('data/m2.csv', index_label = 'date')

    return m2_yoy


def load_re_index():

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(
        t_macro_rlestindex.nyear,
        t_macro_rlestindex.nmonth,
        t_macro_rlestindex.index_gc).statement
    re_index = pd.read_sql(sql, session.bind)

    session.commit()
    session.close()

    dates = []
    for y, m in zip(re_index.nyear.values, re_index.nmonth.values):
        d = monthrange(y, m)[1]
        date = datetime(y, m, d)
        dates.append(date)

    re_index.index = dates
    re_index = re_index.sort_index()

    return re_index


def load_eps_mean():

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_ix_finindex.publishdate, tq_ix_finindex.epscut)\
        .filter(tq_ix_finindex.secode == 2070000005)\
        .statement
    eps_mean = pd.read_sql(
        sql,
        session.bind,
        index_col=['publishdate'],
        parse_dates=['publishdate'])

    eps_mean = eps_mean.sort_index()
    eps_mean = eps_mean.resample('m').last().fillna(method='pad')
    eps_mean = eps_mean.pct_change(12)
    eps_mean = eps_mean.dropna()
    #eps_mean.to_csv('data/eps_mean.csv', index_label = 'date')

    dates = eps_mean.index
    redates = []
    for date in dates:
        if date.month == 12:
            tmp_date = date + timedelta(90)
        elif date.month == 3:
            tmp_date = date + timedelta(30)
        elif date.month == 6:
            tmp_date = date + timedelta(60)
        elif date.month == 9:
            tmp_date = date + timedelta(30)

        redates.append(tmp_date)

    today = datetime.today()
    if redates[-1] > today:
        redate[-1] = today
    eps_mean.index = redates

    return eps_mean


def load_ngdp_yoy():

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(
        t_macro_qgdp.nyear,
        t_macro_qgdp.nmonth,
        t_macro_qgdp.value).statement
    ngdp = pd.read_sql(sql, session.bind)

    dates = []
    for y, m in zip(ngdp.nyear.values, ngdp.nmonth.values):
        d = monthrange(y, m)[1]
        date = datetime(y, m, d)
        dates.append(date)

    ngdp.index = dates
    ngdp = ngdp.sort_index()
    ngdp = ngdp.resample('m').last().fillna(method='pad')
    ngdp['ngdp_yoy'] = ngdp.value.pct_change(12)
    ngdp = ngdp.dropna()

    return ngdp


def load_10Y_bond_ytm():

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(
        tq_qt_cbdindex.tradedate,
        tq_qt_cbdindex.avgmktcapmatyield,
        ).filter(tq_qt_cbdindex.secode == 2070011252).statement
    ytm = pd.read_sql(sql, session.bind, index_col = ['tradedate'], parse_dates = ['tradedate']) 
    ytm.columns = ['ytm']

    ytm = ytm.sort_index()
    #ytm.to_csv('data/caihui_bond.csv', index_label = 'date')

    return ytm


def load_social_finance():
    pass 
