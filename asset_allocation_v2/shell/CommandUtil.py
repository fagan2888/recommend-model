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
@click.option('--path', 'optpath', default=True, help='file path')
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



    portfolio_alloc_data = []
    portfolio_asset_data = []
    portfolio_argv_data  = []


    portfolio_alloc_columns = ['globalid', 'ra_name', 'ra_portfolio_id', 'ra_ratio_id', 'ra_risk', 'ra_type']
    portfolio_asset_columns = ['ra_portfolio_id', 'ra_asset_id', 'ra_asset_name', 'ra_asset_type', 'ra_pool_id']
    portfolio_argv_columns  = ['ra_portfolio_id', 'ra_key', 'ra_value']


    portfolio_alloc_index = ['globalid']
    portfolio_asset_index = ['ra_portfolio_id', 'ra_asset_id']
    portfolio_argv_index  = ['ra_portfolio_id', 'ra_key']


    for k, v in df.groupby(['risk']):

        portfolio_id = v['ra_portfolio_id'].unique().item()
        risk = v['risk'].unique().item()
        portfolio_id_num = portfolio_id.strip().split('.')[1]
        portfolio_risk_id = portfolio_id.replace(portfolio_id_num, str(string.atoi(portfolio_id_num) + int(risk * 10) % 10))
        highlow_id_num = highlow_id.strip().split('.')[1]
        highlow_risk_id = highlow_id.replace(highlow_id_num, str(string.atoi(highlow_id_num) + int(risk * 10) % 10))


        portfolio_alloc_data.append([portfolio_risk_id, portfolio_name, portfolio_id, highlow_risk_id, risk, 9])


        for i in range(0, len(v)):
            record = v.iloc[i]
            asset_id = record['asset_id']
            pool_id  = record['pool_id']
            asset_name = find_asset_name(asset_id)
            portfolio_asset_data.append([portfolio_risk_id, asset_id, asset_name, 0, pool_id])


        portfolio_argv_data = []
        for col in v.columns:
            key = col.strip()
            if key.startswith('portfolio'):
                value = str(v[col].unique().item()).strip()
                value = value if not value == 'nan' else ''
                portfolio_argv_data.append([portfolio_risk_id, key, value])


    portfolio_alloc_df =  pd.DataFrame(portfolio_alloc_data, columns = portfolio_alloc_columns)
    portfolio_alloc_df =  portfolio_alloc_df.set_index(portfolio_alloc_index)

    portfolio_asset_df = pd.DataFrame(portfolio_asset_data, columns = portfolio_asset_columns)
    portfolio_asset_df = portfolio_asset_df.set_index(portfolio_asset_index)

    portfolio_argv_df  = pd.DataFrame(portfolio_argv_data, columns = portfolio_argv_columns)
    portfolio_argv_df  = portfolio_argv_df.set_index(portfolio_argv_index)

    #print portfolio_alloc_df
    #print portfolio_asset_df
    #print portfolio_argv_df

    df_new = portfolio_alloc_df
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(portfolio_alloc_t.c.ra_portfolio_id.in_(v['ra_portfolio_id'].ravel()))
    df_old = pd.read_sql(s, db, index_col = [df_new.index.name])
    database.batch(db, portfolio_alloc_t, df_new, df_old)


    df_new = portfolio_asset_df
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(portfolio_asset_t.c.ra_portfolio_id.in_(df_new.index.get_level_values(0).tolist()))
    #s = s.where(portfolio_asset_t.c.ra_asset_id.in_(df_new.index.get_level_values(1).tolist()))
    df_old = pd.read_sql(s, db, index_col = df_new.index.names)
    database.batch(db, portfolio_asset_t, df_new, df_old)


    df_new = portfolio_argv_df
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(portfolio_argv_t.c.ra_portfolio_id.in_(df_new.index.get_level_values(0).tolist()))
    #s = s.where(portfolio_argv_t.c.ra_key.in_(df_new.index.get_level_values(1).tolist()))
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

    highlow_alloc_data = []
    highlow_asset_data = []
    highlow_argv_data  = []

    highlow_alloc_columns = ['globalid', 'mz_name', 'mz_type', 'mz_highlow_id', 'mz_risk', 'mz_algo', 'mz_markowitz_id']
    highlow_asset_columns = ['mz_highlow_id', 'mz_asset_id', 'mz_asset_name', 'mz_asset_type', 'mz_origin_id', 'mz_riskmgr_id', 'mz_pool_id']
    highlow_argv_columns  = ['mz_highlow_id', 'mz_key', 'mz_value']

    highlow_alloc_index = ['globalid']
    highlow_asset_index = ['mz_highlow_id', 'mz_asset_id']
    highlow_argv_index  = ['mz_highlow_id', 'mz_key']


    for k, v in df.groupby(['risk']):

        highlow_id = v['mz_highlow_id'].unique().item()
        risk = v['risk'].unique().item()
        highlow_id_num = highlow_id.strip().split('.')[1]
        highlow_risk_id = highlow_id.replace(highlow_id_num, str(string.atoi(highlow_id_num) + int(risk * 10) % 10))
        highlow_name = v['mz_highlow_name'].unique().item()
        markowitz_id_num = markowitz_id.strip().split('.')[1]
        markowitz_risk_id = markowitz_id.replace(markowitz_id_num, str(string.atoi(markowitz_id_num) + int(risk * 10) % 10))
        highlow_algo = v['mz_highlow_algo'].unique().item()


        highlow_alloc_data.append([highlow_risk_id, highlow_name, 9, highlow_id, risk, highlow_algo, markowitz_risk_id])


        for i in range(0, len(v)):
            record = v.iloc[i]
            asset_id = record['asset_id']
            pool_id  = record['pool_id']
            riskmgr_id  = record['riskmgr_id']
            asset_name = find_asset_name(asset_id)
            highlow_asset_data.append([highlow_risk_id, asset_id, asset_name, 0, markowitz_id, riskmgr_id, pool_id])



        data = []
        for col in v.columns:
            key = col.strip()
            if key.startswith('highlow'):
                value = str(v[col].unique().item()).strip()
                value = value if not value == 'nan' else ''
                highlow_argv_data.append([highlow_risk_id, key, value])

        #print argv_df


    highlow_alloc_df =  pd.DataFrame(highlow_alloc_data, columns = highlow_alloc_columns)
    highlow_alloc_df =  highlow_alloc_df.set_index(highlow_alloc_index)

    highlow_asset_df = pd.DataFrame(highlow_asset_data, columns = highlow_asset_columns)
    highlow_asset_df = highlow_asset_df.set_index(highlow_asset_index)

    highlow_argv_df  = pd.DataFrame(highlow_argv_data, columns = highlow_argv_columns)
    highlow_argv_df  = highlow_argv_df.set_index(highlow_argv_index)

    #print highlow_alloc_df
    #print highlow_asset_df
    #print highlow_argv_df

    df_new = highlow_alloc_df
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(highlow_alloc_t.c.mz_highlow_id.in_(df_new['mz_highlow_id'].ravel()))
    df_old = pd.read_sql(s, db, index_col = [df_new.index.name])
    database.batch(db, highlow_alloc_t, df_new, df_old)


    df_new = highlow_asset_df.fillna('')
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(highlow_asset_t.c.mz_highlow_id.in_(df_new.index.get_level_values(0).tolist()))
    #s = s.where(highlow_asset_t.c.mz_asset_id.in_(df_new.index.get_level_values(1).tolist()))
    df_old = pd.read_sql(s, db, index_col = df_new.index.names)
    database.batch(db, highlow_asset_t, df_new, df_old)


    df_new = highlow_argv_df
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(highlow_argv_t.c.mz_highlow_id.in_(df_new.index.get_level_values(0).tolist()))
    #s = s.where(highlow_argv_t.c.mz_key.in_(df_new.index.get_level_values(1).tolist()))
    df_old = pd.read_sql(s, db, index_col = df_new.index.names)
    database.batch(db, highlow_argv_t, df_new, df_old)


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


    markowitz_alloc_data = []
    markowitz_asset_data = []
    markowitz_argv_data  = []

    markowitz_alloc_columns = ['globalid', 'mz_markowitz_id', 'mz_name', 'mz_type', 'mz_algo', 'mz_risk']
    markowitz_asset_columns = ['mz_markowitz_id', 'mz_asset_id', 'mz_asset_name', 'mz_markowitz_asset_id','mz_markowitz_asset_name', 'mz_asset_type', 'mz_upper_limit', 'mz_lower_limit', 'mz_sum1_limit', 'mz_sum2_limit']
    markowitz_argv_columns  = ['mz_markowitz_id', 'mz_key', 'mz_value']

    markowitz_alloc_index = ['globalid']
    markowitz_asset_index = ['mz_markowitz_id', 'mz_markowitz_asset_id']
    markowitz_argv_index  = ['mz_markowitz_id', 'mz_key']


    for k, v in df.groupby(['risk']):

        markowitz_id = v['mz_markowitz_id'].unique().item()
        risk = v['risk'].unique().item()
        markowitz_id_num = markowitz_id.strip().split('.')[1]
        markowitz_risk_id = markowitz_id.replace(markowitz_id_num, str(string.atoi(markowitz_id_num) + int(risk * 10) % 10))
        markowitz_algo               = v['allocate_algo'].unique().item()

        markowitz_alloc_data.append([markowitz_risk_id, markowitz_id, markowitz_name, 9, markowitz_algo, risk])


        for i in range(0, len(v)):
            record = v.iloc[i]
            asset_id = record['asset_id']
            asset_name = find_asset_name(asset_id)
            sum1  = record['sum1']
            sum2  = record['sum2']
            lower = record['lower']
            upper = record['upper']
            markowitz_asset_data.append([markowitz_risk_id, asset_id, asset_name, asset_id, asset_name, 0, upper, lower ,sum1, sum2])


        for col in v.columns:
            key = col.strip()
            if key.startswith('allocate'):
                value = str(v[col].unique().item()).strip()
                value = value if not value == 'nan' else ''
                markowitz_argv_data.append([markowitz_risk_id, key, value])


    markowitz_alloc_df =  pd.DataFrame(markowitz_alloc_data, columns = markowitz_alloc_columns)
    markowitz_alloc_df =  markowitz_alloc_df.set_index(markowitz_alloc_index)

    markowitz_asset_df = pd.DataFrame(markowitz_asset_data, columns = markowitz_asset_columns)
    markowitz_asset_df = markowitz_asset_df.set_index(markowitz_asset_index)

    markowitz_argv_df  = pd.DataFrame(markowitz_argv_data, columns = markowitz_argv_columns)
    markowitz_argv_df  = markowitz_argv_df.set_index(markowitz_argv_index)

    #print highlow_alloc_df


    df_new = markowitz_alloc_df
    #print df_new
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(markowitz_alloc_t.c.mz_markowitz_id.in_(df_new['mz_markowitz_id'].ravel()))
    #print s.compile(compile_kwargs={"literal_binds": True})
    df_old = pd.read_sql(s, db, index_col = [df_new.index.name])
    database.batch(db, markowitz_alloc_t, df_new, df_old)



    df_new = markowitz_asset_df
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(markowitz_asset_t.c.mz_markowitz_id.in_(df_new.index.get_level_values(0).tolist()))
    #s = s.where(markowitz_asset_t.c.mz_markowitz_asset_id.in_(df_new.index.get_level_values(1).tolist()))
    df_old = pd.read_sql(s, db, index_col = df_new.index.names)
    database.batch(db, markowitz_asset_t, df_new, df_old)


    df_new = markowitz_argv_df
    columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
    s = select(columns)
    s = s.where(markowitz_argv_t.c.mz_markowitz_id.in_(df_new.index.get_level_values(0).tolist()))
    #s = s.where(markowitz_argv_t.c.mz_key.in_(df_new.index.get_level_values(1).tolist()))
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
@click.option('--path', 'optpath', default=True, help='--path id')
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
@click.option('--path', 'optpath', help='--path id')
@click.option('--poolid', 'optpoolid', help='--pool id')
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
@click.option('--fundtype', 'optfundtype', help='fund type')
@click.option('--inpath', 'optinpath', default=True, help='fund code path')
#@click.option('--outpath', 'optoutpath', default=True, help=u'out fund code path')
@click.option('--poolid', 'optpoolid', help='--pool id')
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

    print(final_codes)

    df = pd.DataFrame(final_codes, columns = ['ra_fund_code'])
    df['ra_pool_id'] = optpoolid.strip()
    df = df.set_index(['ra_pool_id', 'ra_fund_code'])


    db = database.connection('asset')
    t = Table('ra_pool_sample', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]

    for pool_id in set(df.index.get_level_values(0)):
        s = select(columns, (t.c.ra_pool_id == pool_id))
        df_old = pd.read_sql(s, db, index_col=['ra_pool_id', 'ra_fund_code'])
        print(df)
        database.batch(db, t, df, df_old, timestamp=True)



@util.command()
@click.option('--from', 'optfrom', default=True, help='--from id')
@click.option('--to', 'optto', default=True, help='--to id')
@click.option('--name', 'optname', default=True, help='name')
@click.pass_context
def cp(ctx, optfrom, optto, optname):
    pass


@util.command()
@click.option('--highlow-id', 'optid', default=True, help='highlow id')
@click.option('--date', 'optdate', default=True, help='date')
@click.pass_context
def export_highlow_pos(ctx, optid, optdate):

    db = database.connection('asset')

    t = Table('mz_highlow_pos', MetaData(bind=db), autoload=True)

    columns = [
           t.c.mz_asset_id,
           t.c.mz_ratio
    ]

    s = select(columns).where(t.c.mz_date == optdate.strip()).where(t.c.mz_highlow_id == optid.strip())
    df = pd.read_sql(s, db, index_col = ['mz_asset_id'])

    df.to_csv('pos.csv')
