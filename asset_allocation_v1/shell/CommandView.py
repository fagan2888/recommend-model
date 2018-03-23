#coding=utf8


import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import Const
import DBData

from sqlalchemy import MetaData, Table, select, func
from db import *
from db import asset_ra_bl, asset_ra_bl_argv, asset_ra_bl_asset, asset_ra_bl_view, asset_mc_view, asset_mc_view_strength
from CommandMarkowitz import load_wavelet_nav_series, load_nav_series
from heapq import nlargest
from scipy.stats import rankdata

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
        elif view['bl_method'] == 1:
            signal_update_macro(view)


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

    dates = df_new.index.unique()
    df_new = df_new.astype('object')
    for date in dates:
        df_new.loc[date, 'bl_view'] = convert_sharpe_to_view_4(df_new.loc[date, 'bl_view'].values.ravel())

    df_new = df_new.reset_index()
    df_new = df_new.set_index(['bl_date', 'globalid', 'bl_index_id'])

    asset_ra_bl_view.save(viewid, df_new)


## asset with max sharpe ratio get view 1, asset with min sharpe ratio get view -1
def convert_sharpe_to_view(x):
    x = x.astype(float)
    Min = min(x)
    Max = max(x)
    for i in range(len(x)):
        if x[i] == Min:
            x[i] = -1
        elif x[i] == Max:
            x[i] = 1
        else:
            x[i] = 0

    return x

## asset with max sharpe ratio get view 1
def convert_sharpe_to_view_2(x):
    x = x.astype(float)
    Max = max(x)
    for i in range(len(x)):
        if x[i] == Max:
            x[i] = 1
        else:
            x[i] = 0

    return x


## asset with max sharpe ratio get view 2, asset with second largest sharpe ratio get view 1
def convert_sharpe_to_view_3(x):
    x = x.astype(float)
    Max, Smax = nlargest(2, x)
    for i in range(len(x)):
        if x[i] == Max:
            x[i] = 2
        elif x[i] == Smax:
            x[i] = 1
        else:
            x[i] = 0

    return x


def convert_sharpe_to_view_4(x):
    x = x.astype(float)
    rank_x = rankdata(x) - np.median(rankdata(x))

    return rank_x


def cal_wavelet_view(view, indexid, wavenum, trend_lookback, start_date):
    trade_dates = DBData.trade_date_index(start_date = start_date)
    bl_views = []
    for date in trade_dates:
        lookback_date = DBData.trade_date_lookback_index(end_date = date, lookback = int(trend_lookback))
        wavelet_nav_series = load_wavelet_nav_series(indexid, end_date = date, wavelet_filter_num = int(wavenum))
        view_nav_series = wavelet_nav_series.loc[lookback_date].dropna()

        ## view ver 1
        # bl_view = np.sign(view_nav_series.iloc[-1]-view_nav_series.iloc[0])

        ## view ver 2
        # nav_series = load_nav_series(indexid, end_date = date)
        # nav_series = nav_series.loc[view_nav_series.index]
        # view_nav_series = view_nav_series / view_nav_series.iloc[0]
        # nav_series = nav_series / nav_series.iloc[0]
        # bl_view = np.sign(view_nav_series.iloc[-1] - nav_series.iloc[-1])
        wave_ret = (view_nav_series.pct_change().dropna()).mean()
        wave_std = (view_nav_series.pct_change().dropna()).std()
        wave_shape = (wave_ret - Const.rf)/wave_std
        bl_view = wave_shape

        bl_views.append(bl_view)

    df = pd.DataFrame(bl_views, index = trade_dates, columns = ['bl_view'])
    df.index.name = 'bl_date'
    df['globalid'] = view['globalid']
    df['bl_index_id'] = indexid
    # df = df.reset_index()
    # df = df.set_index(['bl_date', 'globalid', 'bl_index_id'])
    return df


def signal_update_macro(view):
    viewid = view['globalid']
    assets = asset_ra_bl_asset.load_assets(id_ = viewid)
    bl_views = []
    for asset in assets:
        mc_view_id = asset_mc_view.get_view_id(asset)
        bl_view = asset_mc_view_strength.load_view_strength(mc_view_id)
        bl_view = np.sign(bl_view)
        bl_view = bl_view.reset_index()
        bl_view.columns = ['bl_date', 'bl_view']
        bl_view['globalid'] = viewid
        bl_view['bl_index_id'] = asset
        bl_views.append(bl_view)

    df_bl_views = pd.concat(bl_views)
    df_bl_views = df_bl_views.set_index(['bl_date', 'globalid', 'bl_index_id'])
    asset_ra_bl_view.save(viewid, df_bl_views)
