#coding=utf8


import sys
sys.path.append('shell')
import click
import pandas as pd

from sqlalchemy import MetaData, Table, select
<<<<<<< e34043447b424bd09c1b7b993078f49cf9579c89
from TimingWavelet import TimingWt
=======
from TimingWavelet import get_filtered_data
>>>>>>> add CommandWavelet
from db import asset_wt_filter, base_ra_index_nav, database


@click.group(invoke_without_command=True)
@click.option('--id', 'optid', help=u'filtered index id to update')
@click.pass_context
<<<<<<< e34043447b424bd09c1b7b993078f49cf9579c89
def filtering(ctx, optid):
=======
def filtering(ctx, optid, optonline):
>>>>>>> add CommandWavelet
    '''filtering group
    '''
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        ctx.invoke(nav, optid=optid)
    else:
        # click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        pass

@filtering.command()
@click.option('--id', 'optid', help=u'ids of fund pool to update')
@click.pass_context
<<<<<<< e34043447b424bd09c1b7b993078f49cf9579c89
def nav(ctx, optid):
=======
def nav(ctx, optid, wavenum, startdate, enddate):
>>>>>>> add CommandWavelet
    ''' calc pool nav and inc
    '''
    if optid is not None:
        filterings = [s.strip() for s in optid.split(',')]
    else:
        filterings = None

    df_filtering = asset_wt_filter.load(filterings)

    with click.progressbar(length=len(df_filtering), label='update signal') as bar:
        for _, filtering in df_filtering.iterrows():
            bar.update(1)
            nav_update(filtering)

def nav_update(filtering):
    ori_data = base_ra_index_nav.load_series(filtering['wt_index_id'])
    wt = TimingWt(ori_data)
    filtered_data = wt.get_filtered_data(ori_data, filtering['wt_filter_num'], \
            filtering['wt_begin_date'])

    df_new = database.number_format(filtered_data, wt_nav = 2, wt_inc = 6, \
            precision=2)
    df_new['wt_filter_id'] = filtering['globalid']
    df_new = df_new.set_index(['wt_filter_id', 'wt_date'])

    # 加载旧数据
    db = database.connection('asset')
    t = Table('wt_filter_nav', MetaData(bind=db), autoload=True)
    columns = [
        t.c.wt_filter_id,
        t.c.wt_date,
        t.c.wt_nav,
        t.c.wt_inc,
    ]
    stmt_select = select(columns, (t.c.wt_filter_id == filtering['globalid']))
    df_old = pd.read_sql(stmt_select, db, index_col=['wt_filter_id', 'wt_date'], \
            parse_dates=['wt_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, wt_nav = 2, wt_inc = 6)

    # 更新数据库
    print
    print df_new.head()
    database.batch(db, t, df_new, df_old, timestamp=False)

    #print df_result.head()
=======
    filtered_data = get_filtered_data(ori_data, filtering['wt_filter_num'], \
            filtering['wt_begin_date'])

    df_new = database.number_format(filtered_data, columns=['wt_nav', 'wt_inc'], \
            precision=2)
    df_new['wt_filter_id'] = filtering['globalid']
    print df_new

'''
    # 加载旧数据
    db = database.connection('asset')
    t2 = Table('wt_filtering_nav', MetaData(bind=db), autoload=True)
    columns2 = [
        t2.c.tc_timing_id,
        t2.c.tc_date,
        t2.c.tc_nav,
        t2.c.tc_inc,
    ]
    stmt_select = select(columns2, (t2.c.tc_timing_id == filtering['filter_index_id']))
    df_old = pd.read_sql(stmt_select, db, index_col=['tc_timing_id', 'tc_date'], parse_dates=['tc_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, columns=['tc_nav', 'tc_inc'], precision=2)

    # 更新数据库
    database.batch(db, t2, df_new, df_old, timestamp=False)

    #print df_result.head()
'''
>>>>>>> add CommandWavelet

