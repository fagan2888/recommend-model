#coding=utf8


import sys
sys.path.append('shell')
import click
import pandas as pd

from sqlalchemy import MetaData, Table, select
from TimingWavelet import TimingWt
from db import asset_wt_filter, base_ra_index_nav, database


@click.group(invoke_without_command=True)
@click.option('--id', 'optid', help=u'filtered index id to update')
@click.option('--full/--no-full', 'optfull', default=True, help=u'include all instance')
@click.pass_context
def filtering(ctx, optid, optfull):
    '''filtering group
    '''
    if ctx.invoked_subcommand is None:
        # click.echo('I was invoked without subcommand')
        if not optfull:
            ctx.invoke(nav, optid=optid)
        else:
            df_filtering = asset_wt_filter.load(None)
            for optid in df_filtering['globalid']:
                ctx.invoke(nav, optid = optid)
    else:
        # click.echo('I am about to invoke %s' % ctx.invoked_subcommand)
        pass

@filtering.command()
@click.option('--id', 'optid', help=u'ids of fund pool to update')
@click.pass_context
def nav(ctx, optid):
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
    print ori_data

    wt = TimingWt(ori_data)
    filtered_data = wt.get_filtered_data(ori_data, filtering['wt_filter_num'], \
            filtering['wt_begin_date'])
    filtered_data = filtered_data.fillna(0.0)

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
    #print
    #print df_new.head()
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


def load_nav_series(asset_id, reindex=None, begin_date=None, end_date=None):


    if asset_id.isdigit():
        xtype = int(asset_id) / 10000000
    else:
        xtype = re.sub(r'([\d]+)','',asset_id).strip()


    if xtype == 1:
        #
        # 基金池资产
        #
        asset_id %= 10000000
        (pool_id, category) = (asset_id / 100, asset_id % 100)
        ttype = pool_id / 10000
        sr = asset_ra_pool_nav.load_series(
            pool_id, category, ttype, reindex=reindex, begin_date=begin_date, end_date=end_date)
    elif xtype == 3:
        #
        # 基金池资产
        #
        sr = base_ra_fund_nav.load_series(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
    elif xtype == 4:
        #
        # 修型资产
        #
        sr = asset_rs_reshape_nav.load_series(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
    elif xtype == 12:
        #
        # 指数资产
        #
        sr = base_ra_index_nav.load_series(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
    elif xtype == 'ERI':

        sr = base_exchange_rate_index_nav.load_series(
            asset_id, reindex=reindex, begin_date=begin_date, end_date=end_date)
    else:
        sr = pd.Series()

    return sr
