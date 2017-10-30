#coding=utf8


import string
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import os
import time
import logging
import re
import util_numpy as npu


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func, literal_column
from tabulate import tabulate
from db import database, base_exchange_rate_index, base_ra_index
from util import xdict

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def util(ctx):
    '''
        import, update, cp portfolio highlow markowitz or ra_pool_sample
    '''
    pass


@util.command()
@click.option('--path', 'optpath', default=True, help=u'file path')
@click.pass_context
def imp_portfolio(ctx, optpath):

    all_portfolio_df = pd.read_csv(optpath.strip(), parse_dates = ['start_date', 'end_date'], dtype = {'asset_id':str})

    imp_markowitz(all_portfolio_df)
    imp_highlow(all_portfolio_df)
    imp_portf(all_portfolio_df)


def imp_portf(df):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    portfolio_t = Table('ra_portfolio', metadata, autoload=True)
    portfolio_alloc_t = Table('ra_portfolio_alloc', metadata, autoload=True)
    portfolio_asset_t = Table('ra_portfolio_asset', metadata, autoload=True)
    portfolio_argv_t = Table('ra_portfolio_argv', metadata, autoload=True)


    df = df.copy()

    portfolio_id               = df['ra_portfolio_id'].unique().item()
    portfolio_name             = df['ra_portfolio_name'].unique().item()
    highlow_id                 = df['mz_highlow_id'].unique().item()
    portfolio_df = pd.DataFrame([[portfolio_id, portfolio_name, highlow_id]], columns = ['globalid', 'ra_name', 'ra_ratio_id'])
    portfolio_df['ra_type'] = 9
    portfolio_df = portfolio_df.set_index(['globalid'])
    portfolio_df['ra_algo'] = df['ra_portfolio_algo'].unique().item()
    portfolio_df['ra_persistent'] = 0

    df_new = portfolio_df
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(portfolio_t.c.globalid.in_(df_new.index.tolist()))
    df_old = pd.read_sql(s, db, index_col = df_new.index.names)
    database.batch(db, portfolio_t, df_new, df_old)

    for k, v in df.groupby(['risk']):
        portfolio_id = v['ra_portfolio_id'].unique().item()
        risk = v['risk'].unique().item()
        portfolio_id_num = portfolio_id.strip().split('.')[1]
        portfolio_risk_id = portfolio_id.replace(portfolio_id_num, str(string.atoi(portfolio_id_num) + int(risk * 10) % 10))
        highlow_id_num = highlow_id.strip().split('.')[1]
        highlow_risk_id = highlow_id.replace(highlow_id_num, str(string.atoi(highlow_id_num) + int(risk * 10) % 10))


        portfolio_alloc_df = pd.DataFrame([[portfolio_risk_id, portfolio_name, portfolio_id, highlow_risk_id, risk]], columns = ['globalid', 'ra_name', 'ra_portfolio_id', 'ra_ratio_id', 'ra_risk'])
        portfolio_alloc_df = portfolio_alloc_df.set_index(['globalid'])
        portfolio_alloc_df['ra_type'] = 9


        df_new = portfolio_alloc_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        s = s.where(portfolio_alloc_t.c.globalid.in_(df_new.index.tolist()))
        df_old = pd.read_sql(s, db, index_col = [df_new.index.name])
        database.batch(db, portfolio_alloc_t, df_new, df_old)


        portfolio_asset_df = v[['asset_id', 'pool_id']]
        portfolio_asset_df['ra_portfolio_id'] = portfolio_risk_id

        portfolio_asset_df = portfolio_asset_df.rename(columns = {'asset_id':'ra_asset_id', 'pool_id' : 'ra_pool_id'})

        portfolio_asset_df['ra_asset_type'] = 0

        asset_names = []
        for asset_id in portfolio_asset_df['ra_asset_id']:
            asset_names.append(find_asset_name(asset_id))
        portfolio_asset_df['ra_asset_name'] = asset_names
        portfolio_asset_df = portfolio_asset_df.set_index(['ra_portfolio_id', 'ra_asset_id'])


        df_new = portfolio_asset_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        s = s.where(portfolio_asset_t.c.ra_portfolio_id.in_(df_new.index.get_level_values(0).tolist()))
        s = s.where(portfolio_asset_t.c.ra_asset_id.in_(df_new.index.get_level_values(1).tolist()))
        df_old = pd.read_sql(s, db, index_col = df_new.index.names)
        database.batch(db, portfolio_asset_t, df_new, df_old)

        data = []
        for col in v.columns:
            key = col.strip()
            if key.startswith('portfolio'):
                value = str(v[col].unique().item()).strip()
                value = value if not value == 'nan' else ''
                data.append([portfolio_risk_id, key, value])

        argv_df = pd.DataFrame(data, columns = ['ra_portfolio_id', 'ra_key', 'ra_value'])
        argv_df = argv_df.set_index(['ra_portfolio_id', 'ra_key'])

        df_new = argv_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        s = s.where(portfolio_argv_t.c.ra_portfolio_id.in_(df_new.index.get_level_values(0).tolist()))
        s = s.where(portfolio_argv_t.c.ra_key.in_(df_new.index.get_level_values(1).tolist()))
        df_old = pd.read_sql(s, db, index_col = df_new.index.names)
        database.batch(db, portfolio_argv_t, df_new, df_old)



def imp_highlow(df):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    highlow_t = Table('mz_highlow', metadata, autoload=True)
    highlow_alloc_t = Table('mz_highlow_alloc', metadata, autoload=True)
    highlow_asset_t = Table('mz_highlow_asset', metadata, autoload=True)
    highlow_argv_t = Table('mz_highlow_argv', metadata, autoload=True)

    df = df.copy()

    highlow_id                 = df['mz_highlow_id'].unique().item()
    highlow_name               = df['mz_highlow_name'].unique().item()
    markowitz_id               = df['mz_markowitz_id'].unique().item()
    highlow_df = pd.DataFrame([[highlow_id, highlow_name, markowitz_id]], columns = ['globalid', 'mz_name', 'mz_markowitz_id'])
    highlow_df['mz_type'] = 9
    highlow_df = highlow_df.set_index(['globalid'])
    highlow_df['mz_algo'] = df['mz_highlow_algo'].unique().item()
    highlow_df['mz_persistent'] = 0


    df_new = highlow_df
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(highlow_t.c.globalid.in_(df_new.index.tolist()))
    df_old = pd.read_sql(s, db, index_col = df_new.index.names)
    database.batch(db, highlow_t, df_new, df_old)


    for k, v in df.groupby(['risk']):

        highlow_id = v['mz_highlow_id'].unique().item()
        risk = v['risk'].unique().item()
        highlow_id_num = highlow_id.strip().split('.')[1]
        highlow_risk_id = highlow_id.replace(highlow_id_num, str(string.atoi(highlow_id_num) + int(risk * 10) % 10))
        highlow_name = v['mz_highlow_name'].unique().item()
        markowitz_id_num = markowitz_id.strip().split('.')[1]
        markowitz_risk_id = markowitz_id.replace(markowitz_id_num, str(string.atoi(markowitz_id_num) + int(risk * 10) % 10))


        highlow_alloc_df = pd.DataFrame([[highlow_risk_id, highlow_name, highlow_id, risk, markowitz_risk_id]], columns = ['globalid', 'mz_name', 'mz_highlow_id', 'mz_risk', 'mz_markowitz_id'])
        highlow_alloc_df = highlow_alloc_df.set_index(['globalid'])
        highlow_alloc_df['mz_type'] = 9
        #highlow_alloc_df['mz_high_id'] = ''
        #highlow_alloc_df['mz_low_id'] = ''


        df_new = highlow_alloc_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        s = s.where(highlow_alloc_t.c.globalid.in_(df_new.index.tolist()))
        df_old = pd.read_sql(s, db, index_col = [df_new.index.name])
        database.batch(db, highlow_alloc_t, df_new, df_old)


        highlow_asset_df = v[['asset_id', 'riskmgr_id', 'pool_id']]
        highlow_asset_df['mz_highlow_id'] = highlow_risk_id

        highlow_asset_df = highlow_asset_df.rename(columns = {'asset_id':'mz_asset_id', 'riskmgr_id':'mz_riskmgr_id', 'pool_id' : 'mz_pool_id'})

        highlow_asset_df['mz_asset_type'] = 0
        highlow_asset_df['mz_highlow_id'] = highlow_risk_id

        asset_names = []
        for asset_id in highlow_asset_df['mz_asset_id']:
            asset_names.append(find_asset_name(asset_id))
        highlow_asset_df['mz_asset_name'] = asset_names
        highlow_asset_df['mz_highlow_id'] =  highlow_risk_id
        highlow_asset_df = highlow_asset_df.set_index(['mz_highlow_id', 'mz_asset_id'])
        highlow_asset_df['mz_origin_id'] = markowitz_id


        df_new = highlow_asset_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        s = s.where(highlow_asset_t.c.mz_highlow_id.in_(df_new.index.get_level_values(0).tolist()))
        s = s.where(highlow_asset_t.c.mz_asset_id.in_(df_new.index.get_level_values(1).tolist()))
        df_old = pd.read_sql(s, db, index_col = df_new.index.names)
        database.batch(db, highlow_asset_t, df_new, df_old)


        data = []
        for col in v.columns:
            key = col.strip()
            if key.startswith('highlow'):
                value = str(v[col].unique().item()).strip()
                value = value if not value == 'nan' else ''
                data.append([highlow_risk_id, key, value])

        argv_df = pd.DataFrame(data, columns = ['mz_highlow_id', 'mz_key', 'mz_value'])
        argv_df = argv_df.set_index(['mz_highlow_id', 'mz_key'])

        #print argv_df

        df_new = argv_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        s = s.where(highlow_argv_t.c.mz_highlow_id.in_(df_new.index.get_level_values(0).tolist()))
        s = s.where(highlow_argv_t.c.mz_key.in_(df_new.index.get_level_values(1).tolist()))
        df_old = pd.read_sql(s, db, index_col = df_new.index.names)
        database.batch(db, highlow_argv_t, df_new, df_old)

    pass


def imp_markowitz(df):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    markowitz_t = Table('mz_markowitz', metadata, autoload=True)
    markowitz_alloc_t = Table('mz_markowitz_alloc', metadata, autoload=True)
    markowitz_asset_t = Table('mz_markowitz_asset', metadata, autoload=True)
    markowitz_argv_t = Table('mz_markowitz_argv', metadata, autoload=True)


    df = df.copy()

    markowitz_id                 = df['mz_markowitz_id'].unique().item()
    markowitz_name               = df['mz_markowitz_name'].unique().item()
    markowitz_algo               = 0
    markowitz_df = pd.DataFrame([[markowitz_id, markowitz_name, markowitz_algo]], columns = ['globalid', 'mz_name', 'mz_algo'])
    markowitz_df['mz_type'] = 9
    markowitz_df = markowitz_df.set_index(['globalid'])



    df_new = markowitz_df
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(markowitz_t.c.globalid.in_(df_new.index.tolist()))
    df_old = pd.read_sql(s, db, index_col = [df_new.index.name])
    database.batch(db, markowitz_t, df_new, df_old)



    for k, v in df.groupby(['risk']):

        markowitz_id = v['mz_markowitz_id'].unique().item()
        risk = v['risk'].unique().item()
        markowitz_id_num = markowitz_id.strip().split('.')[1]
        markowitz_risk_id = markowitz_id.replace(markowitz_id_num, str(string.atoi(markowitz_id_num) + int(risk * 10) % 10))
        markowitz_algo               = v['allocate_algo'].unique().item()



        markowitz_alloc_df = pd.DataFrame([[markowitz_risk_id, markowitz_id, markowitz_name, markowitz_algo, risk]], columns = ['globalid', 'mz_markowitz_id', 'mz_name', 'mz_algo', 'mz_risk'])
        markowitz_alloc_df = markowitz_alloc_df.set_index(['globalid'])
        markowitz_alloc_df['mz_type'] = 9



        df_new = markowitz_alloc_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        s = s.where(markowitz_alloc_t.c.globalid.in_(df_new.index.tolist()))
        df_old = pd.read_sql(s, db, index_col = [df_new.index.name])
        database.batch(db, markowitz_alloc_t, df_new, df_old)



        markowitz_asset_df = v[['asset_id', 'sum1', 'sum2', 'lower', 'upper']]
        markowitz_asset_df['mz_markowitz_id'] = markowitz_risk_id

        markowitz_asset_df = markowitz_asset_df.rename(columns = {'asset_id':'mz_asset_id', 'sum1' : 'mz_sum1_limit',
                                        'sum2' : 'mz_sum2_limit','upper' : 'mz_upper_limit','lower' : 'mz_lower_limit',})

        markowitz_asset_df['mz_markowitz_asset_id'] = markowitz_asset_df['mz_asset_id']
        asset_names = []
        for asset_id in markowitz_asset_df['mz_asset_id']:
            asset_names.append(find_asset_name(asset_id))
        markowitz_asset_df['mz_asset_name'] = markowitz_asset_df['mz_markowitz_asset_name'] = asset_names
        markowitz_asset_df = markowitz_asset_df.set_index(['mz_markowitz_id', 'mz_markowitz_asset_id'])


        df_new = markowitz_asset_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        s = s.where(markowitz_asset_t.c.mz_markowitz_id.in_(df_new.index.get_level_values(0).tolist()))
        s = s.where(markowitz_asset_t.c.mz_markowitz_asset_id.in_(df_new.index.get_level_values(1).tolist()))
        df_old = pd.read_sql(s, db, index_col = df_new.index.names)
        database.batch(db, markowitz_asset_t, df_new, df_old)


        data = []
        for col in v.columns:
            key = col.strip()
            if key.startswith('allocate'):
                value = str(v[col].unique().item()).strip()
                value = value if not value == 'nan' else ''
                data.append([markowitz_risk_id, key, value])

        argv_df = pd.DataFrame(data, columns = ['mz_markowitz_id', 'mz_key', 'mz_value'])
        argv_df = argv_df.set_index(['mz_markowitz_id', 'mz_key'])


        df_new = argv_df
        columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
        s = select(columns)
        s = s.where(markowitz_argv_t.c.mz_markowitz_id.in_(df_new.index.get_level_values(0).tolist()))
        s = s.where(markowitz_argv_t.c.mz_key.in_(df_new.index.get_level_values(1).tolist()))
        df_old = pd.read_sql(s, db, index_col = df_new.index.names)
        database.batch(db, markowitz_argv_t, df_new, df_old)


        #print argv_df


def find_asset_name(asset_id):
    if asset_id.strip().isdigit():
        #print int(asset_id)
        asset_id = int(asset_id)
        xtype = asset_id / 10000000
        if 12 == xtype:
            record = base_ra_index.find(asset_id)
            return record[2].strip()
        else:
            return None
    elif asset_id.strip().startswith('ERI'):
        record = base_exchange_rate_index.find(asset_id)
        return record[2].strip()
    else:
        return None
    #flag = asset_id.strip().split('.')[0]



@util.command()
@click.option('--path', 'optpath', default=True, help=u'--path id')
@click.pass_context
def import_pool_sample(ctx, optpath):


    df = pd.read_csv(optpath.strip(), index_col = ['ra_pool_id'])
    db = database.connection('asset')
    t = Table('ra_pool_sample', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]

    for pool_id in set(df.index):
        s = select(columns, (t.c.ra_pool_id == pool_id))
        df_old = pd.read_sql(s, db, index_col=['ra_pool_id'])
        database.batch(db, t, df, df_old, timestamp=True)


@util.command()
@click.option('--path', 'optpath', help=u'--path id')
@click.option('--poolid', 'optpoolid', help=u'--pool id')
@click.pass_context
def import_pool_sample(ctx, optpath, optpoolid):


    df = pd.read_csv(optpath.strip(), index_col = ['ra_pool_id'])

    db = database.connection('asset')
    t = Table('ra_pool_sample', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]

    for pool_id in set(df.index):
        s = select(columns, (t.c.ra_pool_id == pool_id))
        df_old = pd.read_sql(s, db, index_col=['ra_pool_id'])
        database.batch(db, t, df, df_old, timestamp=True)


@util.command()
@click.option('--fundtype', 'optfundtype', help=u'fund type')
@click.option('--inpath', 'optinpath', default=True, help=u'fund code path')
#@click.option('--outpath', 'optoutpath', default=True, help=u'out fund code path')
@click.option('--poolid', 'optpoolid', help=u'--pool id')
@click.pass_context
def filter_fund_by_list_to_pool_sample(ctx, optfundtype, optinpath, optpoolid):


    df = pd.read_csv(optinpath.strip())
    codes_set = set()
    for code in df['codes'].values:
        codes_set.add('%06d' % int(code))

    fund_type = int(optfundtype.strip())

    db = database.connection('base')
    t = Table('ra_fund', MetaData(bind=db), autoload=True)
    columns = [
                t.c.ra_code,
                ]
    s = select(columns)
    s = s.where(t.c.ra_type == fund_type)
    df = pd.read_sql(s, db)

    final_codes = []
    for code in df['ra_code'].values:
        code = '%06d' % int(code)
        if code in codes_set:
            final_codes.append(code)

    print final_codes

    df = pd.DataFrame(final_codes, columns = ['ra_fund_code'])
    df['ra_pool_id'] = optpoolid.strip()
    df = df.set_index(['ra_pool_id', 'ra_fund_code'])


    db = database.connection('asset')
    t = Table('ra_pool_sample', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]

    for pool_id in set(df.index.get_level_values(0)):
        s = select(columns, (t.c.ra_pool_id == pool_id))
        df_old = pd.read_sql(s, db, index_col=['ra_pool_id', 'ra_fund_code'])
        print df
        database.batch(db, t, df, df_old, timestamp=True)



@util.command()
@click.option('--from', 'optfrom', default=True, help=u'--from id')
@click.option('--to', 'optto', default=True, help=u'--to id')
@click.option('--name', 'optname', default=True, help=u'name')
@click.pass_context
def cp(ctx, optfrom, optto, optname):
    pass
