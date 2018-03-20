#coding=utf8


import pdb
import getopt
import string
import json
import os
import sys
import logging
sys.path.append('shell')
import click
import config
import pandas as pd
import numpy as np
import os
import time
import DFUtil
import util_numpy as npu


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from tabulate import tabulate
from db import *
from util.xdebug import dd


logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def exrindex(ctx):
    '''exhange rate index
    '''
    pass




@exrindex.command()
@click.option('--full/--no-full', 'optfull', default=True, help=u'include all instance')
@click.option('--id', 'optid', help=u'specify cnyindex id')
@click.pass_context
def cny(ctx, optfull, optid):


    db = database.connection('base')
    metadata = MetaData(bind=db)
    exchange_rate_index_t = Table('exchange_rate_index', metadata, autoload=True)
    exchange_rate_index_nav_t = Table('exchange_rate_index_nav', metadata, autoload=True)
    ra_index_nav_t = Table('ra_index_nav', metadata, autoload=True)
    caihui_exchange_rate_t = Table('caihui_exchange_rate', metadata, autoload=True)


    exchange_rate_index_columns = [
        exchange_rate_index_t.c.globalid,
        exchange_rate_index_t.c.eri_pcur,
        exchange_rate_index_t.c.eri_excur,
        exchange_rate_index_t.c.eri_pricetype,
        exchange_rate_index_t.c.eri_datasource,
        exchange_rate_index_t.c.eri_index_id,
        exchange_rate_index_t.c.eri_code,
        exchange_rate_index_t.c.eri_name,

    ]

    ra_index_nav_columns = [

        ra_index_nav_t.c.ra_date,
        ra_index_nav_t.c.ra_open,
        ra_index_nav_t.c.ra_high,
        ra_index_nav_t.c.ra_low,
        ra_index_nav_t.c.ra_nav,
        ra_index_nav_t.c.ra_volume,
        ra_index_nav_t.c.ra_amount,
        ra_index_nav_t.c.ra_nav_date,
        ra_index_nav_t.c.ra_mask,
    ]


    caihui_exchange_rate_columns = [

        caihui_exchange_rate_t.c.cer_exchangeprice,
        caihui_exchange_rate_t.c.cer_tradedate,

    ]


    exchange_rate_index_df = None

    if optid is not None:
        s = select(exchange_rate_index_columns).where(exchange_rate_index_t.c.globalid == optid.strip())
        exchange_rate_index_df = pd.read_sql(s, db, index_col = ['globalid'])
    elif optfull:
        s = select(exchange_rate_index_columns)
        exchange_rate_index_df = pd.read_sql(s, db, index_col = ['globalid'])


    records = []
    for i in range(0, len(exchange_rate_index_df)):
        record = exchange_rate_index_df.iloc[i]
        records.append([record.name, record['eri_pcur'], record['eri_excur'], record['eri_pricetype'], record['eri_datasource'], record['eri_index_id'].strip(), record['eri_code'].strip(), record['eri_name'].strip()])


    with click.progressbar(records) as bar:
            for record in bar:
                logger.debug("%s", str(record))


                s = select(caihui_exchange_rate_columns)
                s = s.where(caihui_exchange_rate_t.c.cer_pcur == record[1])
                s = s.where(caihui_exchange_rate_t.c.cer_excur == record[2])
                s = s.where(caihui_exchange_rate_t.c.cer_pricetype == record[3])
                s = s.where(caihui_exchange_rate_t.c.cer_datasource == record[4])

                caihui_exchange_rate_df = pd.read_sql(s, db, index_col = ['cer_tradedate'], parse_dates = ['cer_tradedate'])


                s = select(ra_index_nav_columns)
                s = s.where(ra_index_nav_t.c.ra_index_id == record[5])
                ra_index_nav_df = pd.read_sql(s, db, index_col = ['ra_date'])


                caihui_exchange_rate_df = caihui_exchange_rate_df.reindex(ra_index_nav_df.index)
                caihui_exchange_rate_df = caihui_exchange_rate_df.fillna(method = 'pad')

                exchange_rate_index_nav_df = pd.DataFrame([])
                exchange_rate_index_nav_df['eri_open'] = ra_index_nav_df['ra_open'] * caihui_exchange_rate_df['cer_exchangeprice']
                exchange_rate_index_nav_df['eri_high'] = ra_index_nav_df['ra_high'] * caihui_exchange_rate_df['cer_exchangeprice']
                exchange_rate_index_nav_df['eri_low'] = ra_index_nav_df['ra_low'] * caihui_exchange_rate_df['cer_exchangeprice']
                exchange_rate_index_nav_df['eri_nav'] = ra_index_nav_df['ra_nav'] * caihui_exchange_rate_df['cer_exchangeprice']
                exchange_rate_index_nav_df['eri_inc'] = exchange_rate_index_nav_df['eri_nav'].pct_change().fillna(0.0)
                exchange_rate_index_nav_df['eri_index_code'] = record[6]
                exchange_rate_index_nav_df['eri_name'] = record[7]
                exchange_rate_index_nav_df['eri_date'] = ra_index_nav_df.index
                exchange_rate_index_nav_df['eri_nav_date'] = ra_index_nav_df['ra_nav_date']
                exchange_rate_index_nav_df['eri_mask'] = ra_index_nav_df['ra_mask']
                exchange_rate_index_nav_df['eri_index_id'] = record[0]
                exchange_rate_index_nav_df['eri_volume'] = ra_index_nav_df['ra_volume']
                exchange_rate_index_nav_df['eri_amount'] = ra_index_nav_df['ra_amount']
                exchange_rate_index_nav_df = exchange_rate_index_nav_df.set_index(['eri_index_id', 'eri_date'])


                df_new = exchange_rate_index_nav_df
                #print df_new.index.get_level_values(1).tolist()

                columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
                s = select(columns)
                s = s.where(exchange_rate_index_nav_t.c.eri_index_id.in_(df_new.index.get_level_values(0).tolist()))
                s = s.where(exchange_rate_index_nav_t.c.eri_date.in_(df_new.index.get_level_values(1).tolist()))
                df_old = pd.read_sql(s, db, index_col = ['eri_index_id', 'eri_date'], parse_dates = ['eri_date'])
                database.batch(db, exchange_rate_index_nav_t, df_new, df_old)


                exchange_rate_index_nav_df = exchange_rate_index_nav_df.reset_index()
                ra_index_nav_df = pd.DataFrame([])
                ra_index_nav_df['ra_index_code'] = exchange_rate_index_nav_df['eri_index_id']
                ra_index_nav_df['ra_date'] = exchange_rate_index_nav_df['eri_date']
                ra_index_nav_df['ra_open'] = exchange_rate_index_nav_df['eri_open']
                ra_index_nav_df['ra_high'] = exchange_rate_index_nav_df['eri_high']
                ra_index_nav_df['ra_low'] = exchange_rate_index_nav_df['eri_low']
                ra_index_nav_df['ra_nav'] = exchange_rate_index_nav_df['eri_nav']
                ra_index_nav_df['ra_inc'] = exchange_rate_index_nav_df['eri_inc']
                ra_index_nav_df['ra_volume'] = exchange_rate_index_nav_df['eri_volume']
                ra_index_nav_df['ra_amount'] = exchange_rate_index_nav_df['eri_amount']
                ra_index_nav_df['ra_nav_date'] = exchange_rate_index_nav_df['eri_nav_date']
                ra_index_nav_df['ra_mask'] = exchange_rate_index_nav_df['eri_mask']
                if exchange_rate_index_nav_df['eri_index_id'].iloc[0] == 'ERI000001':
                    ra_index_nav_df['ra_index_id'] = 120000042
                elif exchange_rate_index_nav_df['eri_index_id'].iloc[0] == 'ERI000002':
                    ra_index_nav_df['ra_index_id'] = 120000043

                ra_index_nav_df = ra_index_nav_df.set_index(['ra_index_id', 'ra_date'])


                df_new = ra_index_nav_df
                columns = [literal_column(c) for c in (df_new.index.names + list(df_new.columns))]
                s = select(columns)
                s = s.where(ra_index_nav_t.c.ra_index_id.in_(df_new.index.get_level_values(0).tolist()))
                s = s.where(ra_index_nav_t.c.ra_date.in_(df_new.index.get_level_values(1).tolist()))
                df_old = pd.read_sql(s, db, index_col = ['ra_index_id', 'ra_date'], parse_dates = ['ra_date'])
                database.batch(db, ra_index_nav_t, df_new, df_old)


    pass






