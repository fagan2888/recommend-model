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
import pywt
from ipdb import set_trace

import config
from db import database, asset_trade_dates, base_ra_index_nav
from db.asset_fundamental import *
from calendar import monthrange
from datetime import datetime, timedelta
from asset import Asset
from trade_date import ATradeDate
logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def view(ctx):
    '''
    macro timing
    '''
    pass


@view.command()
@click.option('--start-date', 'startdate', default='2000-01-01', help='start date to calc')
@click.option('--end-date', 'enddate', default=datetime.today().strftime('%Y-%m-%d'), help='start date to calc')
@click.option('--viewid', 'viewid', default='BL.000002', help='macro timing view id')
@click.option('--index', 'idx', default=None, help='macro timing view id')
@click.option('--wavenum', 'wavenum', default=2, help='macro timing view id')
@click.option('--max_wave_num', 'max_wave_num', default=7, help='macro timing view id')
@click.option('--wave_name', 'wave_name', default='sym4', help='macro timing view id')
@click.pass_context
def wavelet(ctx, startdate, enddate, viewid, idx, wavenum, max_wave_num, wave_name):


    if idx is None:
        idx = ['120000001', '120000002', '120000013', '120000014', '120000015', '120000080' ,'ERI000001', 'ERI000002']
        #idx = ['120000001']
    for _id in idx:
        trade_dates = ATradeDate.trade_date()
        nav = Asset(_id).nav(reindex = trade_dates).fillna(method = 'pad').dropna()
        dates = nav.index[1000:]
        views = []
        for d in dates:
            _tmp_nav = nav[nav.index <= d]
            wave_nav = wavefilter(_tmp_nav, wavenum, wname = wave_name, maxlevel = max_wave_num)
            wave_diff_rolling = wave_nav.diff().rolling(5).mean()
            views.append(wave_diff_rolling[-1])

        view_df = pd.DataFrame(views, index = dates, columns = ['bl_view'])
        view_df[view_df > 0] = 1.0
        view_df[view_df < 0] = -1.0
        view_df.index.name = 'bl_date'
        view_df['globalid'] = viewid
        view_df['bl_index_id'] = _id
        view_df['created_at'] = datetime.now()
        view_df['updated_at'] = datetime.now()
        df_new = view_df.reset_index().set_index(['globalid','bl_date','bl_index_id'])

        print(df_new.tail())

        db = database.connection('asset')
        metadata = MetaData(bind=db)
        t = Table('ra_bl_view', metadata, autoload = True)
        columns = [
            t.c.globalid,
            t.c.bl_date,
            t.c.bl_view,
            t.c.bl_index_id,
            t.c.created_at,
            t.c.updated_at,
        ]
        s = select(columns).where(t.c.globalid == viewid).where(t.c.bl_index_id == _id)
        df_old = pd.read_sql(s, db, index_col = ['globalid', 'bl_date', 'bl_index_id'], parse_dates = ['bl_date'])
        database.batch(db, t, df_new, df_old, timestamp = False)


def wavefilter(data, wavenum, wname = 'sym4', maxlevel = 7):

    if len(data)%2 == 1:
        data = data[1:]

    # Decompose the signal
    c = pywt.wavedec(data, wname, level = maxlevel)
    filter_level = list(np.arange(maxlevel + 1))
    filter_level.pop(wavenum)
    for j in filter_level:
        c[j][:] = 0
    fdata = pywt.waverec(c, wname)
    sr = pd.Series(fdata.ravel(), index = data.index)
    return sr
