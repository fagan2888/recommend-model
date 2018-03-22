#coding=utf8


import getopt
import string
import json
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import LabelAsset
import os
import time
import re
import Const
import DFUtil
import DBData

from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func
from tabulate import tabulate
from db import *
from db import asset_ra_bl, asset_ra_bl_argv, asset_ra_bl_asset, asset_ra_bl_view
from CommandMarkowitz import load_wavelet_nav_series, load_nav_series
from util.xdebug import dd

import traceback, code
from ipdb import set_trace

@click.group(invoke_without_command=True)
@click.option('--id', 'optid', help=u'fund pool id to update')
@click.option('--online/--no-online', 'optonline', default=False, help=u'include online instance')
@click.pass_context
def view(ctx, optid, optonline):
    '''add asset view
    '''
    if ctx.invoked_subcommand is None:
        ctx.invoke(signal, optid=optid, optonline=optonline)
    else:
        pass

@view.command()
@click.option('--id', 'optid', help=u'fund pool id to update')
@click.option('--online/--no-online', 'optonline', default=False, help=u'include online instance')
@click.pass_context
def signal(ctx, optid, optonline):

    if optid is not None:
        views = [s.strip() for s in optid.split(',')]
    else:
        views = None

    xtypes = None
    if optonline == False:
        xtypes = [1]

    df_view = asset_ra_bl.load(views, xtypes)

    # with click.progressbar(length=len(df_view), label='update signal'.ljust(30)) as bar:
    #     for _, view in df_view.iterrows():
    #         bar.update(1)
    #         if view['bl_method'] == 2:
    #             signal_update_wavelet(view)

    for _, view in df_view.iterrows():
        if view['bl_method'] == 2:
            signal_update_wavelet(view)
        else:
            pass


def signal_update_wavelet(view):
    viewid = view['globalid']
    argv = asset_ra_bl_argv.load_argv(id_ = viewid)
    assets = asset_ra_bl_asset.load_assets(id_ = viewid)
    asset_views = []
    with click.progressbar(length=len(assets), label='update bl view'.ljust(30)) as bar:
        for asset in assets:
            asset_view = cal_wavelet_view(view, indexid = asset, **argv)
            asset_views.append(asset_view)
            bar.update(1)

    df_new = pd.concat(asset_views)
    asset_ra_bl_view.save(viewid, df_new)


def cal_wavelet_view(view, indexid, wavenum, trend_lookback, start_date):
    trade_dates = DBData.trade_date_index(start_date = start_date)
    bl_views = []
    for date in trade_dates:
        lookback_date = DBData.trade_date_lookback_index(end_date = date, lookback = int(trend_lookback))
        wavelet_nav_series = load_wavelet_nav_series(indexid, end_date = date, wavelet_filter_num = int(wavenum))
        view_nav_series = wavelet_nav_series.loc[lookback_date].dropna()

        nav_series = load_nav_series(indexid, end_date = date)
        nav_series = nav_series.loc[view_nav_series.index]

        view_nav_series = view_nav_series / view_nav_series.iloc[0]
        nav_series = nav_series / nav_series.iloc[0]
        #bl_view = np.sign(view_nav_series.iloc[-1]-view_nav_series.iloc[0])

        #print view_nav_series.tail(), nav_series.tail()
        #print
        bl_view = np.sign(view_nav_series.iloc[-1] - nav_series.iloc[-1])
        bl_views.append(bl_view)

    df = pd.DataFrame(bl_views, index = trade_dates, columns = ['bl_view'])
    df.index.name = 'bl_date'
    df['globalid'] = view['globalid']
    df['bl_index_id'] = indexid
    df = df.reset_index()
    df = df.set_index(['bl_date', 'globalid', 'bl_index_id'])
    return df













