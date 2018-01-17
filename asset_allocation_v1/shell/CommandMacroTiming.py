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
from db import database, asset_trade_dates, base_ra_index_nav, asset_mc_view
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
@click.option('--start-date', 'startdate', default='2012-07-27', help=u'start date to calc')
@click.option('--end-date', 'enddate', default=datetime.today().strftime('%Y-%m-%d'), help=u'start date to calc')
@click.option('--viewid', 'viewid', default='MC.VW0001', help=u'macro timing view id')
@click.pass_context
def macro_view_update(ctx, startdate, enddate, viewid):
    backtest_interval = pd.date_range(startdate, enddate)
    rev = re_view(backtest_interval)
    irv = ir_view(backtest_interval)
    epsv = eps_view(backtest_interval)

    mv = pd.concat([rev, irv, epsv], 1)
    mv['mv'] = mv['rev'] + mv['irv'] + mv['epsv']
    #mv = mv.loc[:, ['mv']]

    today = datetime.now()
    mv_view_id = np.repeat(viewid, len(mv))
    mv_date = mv.index
    mv_inc = mv.mv.values
    created_at = np.repeat(today, len(mv))
    updated_at = np.repeat(today, len(mv))
    #df_inc_value = np.column_stack([mv_view_id, mv_date, mv_inc, created_at, updated_at])
    #df_inc = pd.DataFrame(df_inc_value, columns = ['mc_view_id', 'mc_date', 'mc_inc', 'created_at', 'updated_at'])
    union_mv = {}
    union_mv['mc_view_id'] = mv_view_id
    union_mv['mc_date'] = mv_date
    union_mv['mc_inc'] = mv_inc
    union_mv['created_at'] = created_at
    union_mv['updated_at'] = updated_at
    union_mv_df = pd.DataFrame(union_mv, columns = ['mc_view_id', 'mc_date', 'mc_inc', 'created_at', 'updated_at'])
    df_new = union_mv_df.set_index(['mc_view_id','mc_date'])

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('mc_view_strength', metadata, autoload = True)
    columns = [
        t.c.mc_view_id,
        t.c.mc_date,
        t.c.mc_inc,
        t.c.created_at,
        t.c.updated_at,
    ]
    s = select(columns, (t.c.mc_view_id == viewid))
    df_old = pd.read_sql(s, db, index_col = ['mc_view_id', 'mc_date'], parse_dates = ['mc_date'])
    database.batch(db, t, df_new, df_old, timestamp = False)

    #union_mv_df.to_sql('mc_view_strength', db, index = True, if_exists = 'append', chunksize = 500)

    #macro_view_update(mv, irv)
    #mv.to_csv('data/mv.csv', index_label = 'date')
    #irv.to_csv('data/irv.csv', index_label = 'date')
    #sz = base_ra_index_nav.load_series('120000016') 

@mt.command()
@click.option('--start-date', 'startdate', default='2012-07-27', help=u'start date to calc')
@click.option('--end-date', 'enddate', default=datetime.today().strftime('%Y-%m-%d'), help=u'start date to calc')
@click.option('--viewid', 'viewid', default='MC.VW0002', help=u'macro timing view id')
@click.pass_context
def bond_view_update(ctx, startdate, enddate, viewid):
    backtest_interval = pd.date_range(startdate, enddate)
    mv = ir_view(backtest_interval)

    today = datetime.now()
    mv_view_id = np.repeat(viewid, len(mv))
    mv_date = mv.index
    mv_inc = mv.irv.values
    created_at = np.repeat(today, len(mv))
    updated_at = np.repeat(today, len(mv))
    #df_inc_value = np.column_stack([mv_view_id, mv_date, mv_inc, created_at, updated_at])
    #df_inc = pd.DataFrame(df_inc_value, columns = ['mc_view_id', 'mc_date', 'mc_inc', 'created_at', 'updated_at'])
    union_mv = {}
    union_mv['mc_view_id'] = mv_view_id
    union_mv['mc_date'] = mv_date
    union_mv['mc_inc'] = mv_inc
    union_mv['created_at'] = created_at
    union_mv['updated_at'] = updated_at
    union_mv_df = pd.DataFrame(union_mv, columns = ['mc_view_id', 'mc_date', 'mc_inc', 'created_at', 'updated_at'])
    df_new = union_mv_df.set_index(['mc_view_id', 'mc_date'])

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('mc_view_strength', metadata, autoload = True)
    columns = [
        t.c.mc_view_id,
        t.c.mc_date,
        t.c.mc_inc,
        t.c.created_at,
        t.c.updated_at,
    ]
    s = select(columns, (t.c.mc_view_id == viewid))
    df_old = pd.read_sql(s, db, index_col = ['mc_view_id', 'mc_date'], parse_dates = ['mc_date'])
    database.batch(db, t, df_new, df_old, timestamp = False)


def re_view(bt_int):
    repy = load_re_price_yoy()
    m1 = load_m1_yoy()

    repy['repy_diff'] = repy['repy'].diff(1)
    m1['m1_diff'] = m1['m1'].rolling(6).mean().diff(1)
    repy = repy.dropna()
    m1 = m1.dropna()

    rev = pd.merge(repy, m1, left_index = True, right_index = True)

    re_views = []
    rev = rev.reindex(bt_int).fillna(method = 'pad')
    for day in bt_int:
        repy_diff = rev.loc[day, 'repy_diff']
        m1_diff = rev.loc[day, 'm1_diff']

        if (repy_diff > 0) and (m1_diff > 0):
            re_views.append(-3)
        elif (repy_diff < 0) and (m1_diff < 0):
            re_views.append(4)
#        elif (repy_diff > 0) and (m1_diff < 0):
#            re_views.append(0)
        else:
            re_views.append(0)

    rev_res = pd.DataFrame(data = re_views, index = bt_int, columns = ['rev'])

    return rev_res


def ir_view(bt_int):

    ytm = load_10Y_bond_ytm()
    sf = load_social_finance()
    m2 = load_m2_value()

    sf_m2 = pd.merge(sf, m2, left_index = True, right_index = True, how = 'inner')
    sf_m2['sf_m2'] = (sf_m2['sf'] - sf_m2['m2']).diff(12)
    sf_m2['sf_m2_diff'] = sf_m2['sf_m2'].rolling(12).mean().diff().dropna()
    #sf_m2.to_csv('sf_m2.csv', index_label = 'date')

    ytm = ytm.resample('d').last().fillna(method = 'pad')
    ytm['ytm_diff'] = ytm.diff(20).dropna()
    ir = pd.merge(sf_m2, ytm, left_index = True, right_index = True, how = 'outer').fillna(method = 'pad')
    ir = ir.dropna()
    #ir = sf_m2.join(ytm)
    #ir.to_csv('data/sf_m2.csv', index_label = 'date')

    ir_views = []
    ir = ir.reindex(bt_int).fillna(method = 'pad')
    for day in bt_int:
        sf_m2_diff = ir.loc[day, 'sf_m2_diff']
        ytm_diff = ir.loc[day, 'ytm_diff']
        if (sf_m2_diff > 0) and (ytm_diff > 0):
            ir_views.append(-1)
        elif (sf_m2_diff < 0) and (ytm_diff < 0):
            ir_views.append(2)
        else:
            ir_views.append(0)

    irv = pd.DataFrame(data = ir_views, index = bt_int, columns = ['irv'])

    return irv


def eps_view(bt_int):
    eps_mean = load_eps_mean()
    ngdp = load_ngdp_yoy()
    #ngdp = load_ngdp_yoy()
    epsv = eps_mean.resample('d').last().fillna(method = 'pad')
    epsv['epsv'] = np.sign(epsv.epscut)*3
    epsv = epsv.loc[:, ['epsv']]
    epsv = epsv.reindex(bt_int).fillna(method = 'pad').fillna(0.0)

    return epsv


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

    dates = m1_yoy.index
    redates = []
    for day in dates:
        redates.append(day + timedelta(18))

   # today = datetime.today()
   # if redates[-1] > today:
   #     redates[-1] = today
    m1_yoy.index = redates

    m1_yoy = m1_yoy.loc[:, ['growthrate_m1']]
    m1_yoy.columns = ['m1']

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

    dates = m2_yoy.index
    redates = []
    for day in dates:
        redates.append(day + timedelta(15))

   # today = datetime.today()
   # if redates[-1] > today:
   #     redates[-1] = today
    m2_yoy.index = redates

#    m2_yoy.to_csv('data/m2.csv', index_label = 'date')

    return m2_yoy


def load_m2_value():

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(
        t_macro_msupply.nyear,
        t_macro_msupply.nmonth,
        t_macro_msupply.value_m2).statement
    m2_value = pd.read_sql(sql, session.bind)

    session.commit()
    session.close()

    dates = []
    for y, m in zip(m2_value.nyear.values, m2_value.nmonth.values):
        d = monthrange(y, m)[1]
        date = datetime(y, m, d)
        dates.append(date)

    m2_value.index = dates
    m2_value = m2_value.sort_index()
    m2_value = m2_value.dropna()
#    m2_value.to_csv('data/m2.csv', index_label = 'date')

    dates = m2_value.index
    redates = []
    for day in dates:
        redates.append(day + timedelta(15))

   # today = datetime.today()
   # if redates[-1] > today:
   #     redates[-1] = today
    m2_value.index = redates

    m2_value = m2_value.loc[:, ['value_m2']]
    m2_value.columns = ['m2']

    return m2_value


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

    session.commit()
    session.close()

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

   # today = datetime.today()
   # if redates[-1] > today:
   #     redate[-1] = today
    eps_mean.index = redates
    eps_mean.to_csv('data/eps_mean.csv', index_label = 'date')

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

    session.commit()
    session.close()

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

    dates = ngdp.index
    redates = []
    for day in dates:
        redates.append(dates + timedelta(15))
    ngdp.index = redates
    ngdp.to_csv('data/ngdp.csv', index_label = 'date')

    return ngdp


def load_10Y_bond_ytm():

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(
        tq_qt_cbdindex.tradedate,
        tq_qt_cbdindex.avgmktcapmatyield,
    ).filter(tq_qt_cbdindex.secode == 2070011252).statement

    ytm = pd.read_sql(
        sql,
        session.bind,
        index_col=['tradedate'],
        parse_dates=['tradedate'])

    session.commit()
    session.close()

    ytm.columns = ['ytm']

    ytm = ytm.sort_index()
    #ytm.to_csv('data/caihui_bond.csv', index_label = 'date')

    return ytm


def load_social_finance():

    engine = database.connection('wind')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(
        mc_social_finance.mc_sf_date,
        mc_social_finance.mc_sf_value,
    ).filter(mc_social_finance.globalid == 'MC.SF0001').statement


    sf = pd.read_sql(
        sql,
        session.bind,
        index_col=['mc_sf_date'],
        parse_dates=['mc_sf_date'])

    session.commit()
    session.close()
    sf = sf.sort_index()

    dates = sf.index
    redates = []
    for day in dates:
        redates.append(day + timedelta(15))

   # today = datetime.today()
   # if redates[-1] > today:
   #     redates[-1] = today
    sf.index = redates

    sf.columns = ['sf']

    return sf


def load_re_price_yoy():

    engine = database.connection('wind')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(
        mc_real_estate.mc_re_date,
        mc_real_estate.mc_re_value,
        ).filter(mc_real_estate.globalid == 'MC.RE0001').statement

    repy = pd.read_sql(
        sql,
        session.bind,
        index_col = ['mc_re_date'],
        parse_dates = ['mc_re_date'],
        )
    session.commit()
    session.close()
    repy = repy.sort_index()

    dates = repy.index
    redates= []
    for day in dates:
        redates.append(day + timedelta(18))

   # today = datetime.today()
   # if redates[-1] > today:
   #     redates[-1] = today
    repy.index = redates

    repy.columns = ['repy']

    return repy
