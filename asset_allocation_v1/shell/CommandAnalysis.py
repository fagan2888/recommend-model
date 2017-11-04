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
def analysis(ctx):
    '''
        analysis something
    '''
    pass


@analysis.command()
@click.option('--portfolio-id', 'optid', default=True, help=u'portfolio_id')
@click.option('--start-date', 'optstartdate', default=None, help=u'portfolio pos startdate')
@click.option('--end-date', 'optenddate', default=None, help=u'portfolio pos endate')
@click.pass_context
def portfolio_turnover_freq(ctx, optid, optstartdate, optenddate):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    portfolio_alloc_t = Table('ra_portfolio_alloc', metadata, autoload=True)
    portfolio_pos_t = Table('ra_portfolio_pos', metadata, autoload=True)

    alloc_columns = [
                    portfolio_alloc_t.c.globalid,
            ]

    s = select(alloc_columns).where(portfolio_alloc_t.c.ra_portfolio_id == optid.strip())

    alloc_df = pd.read_sql(s, db)

    pos_columns = [
                portfolio_pos_t.c.ra_date,
                portfolio_pos_t.c.ra_fund_code,
                portfolio_pos_t.c.ra_fund_ratio,
            ]

    for globalid in alloc_df['globalid'].ravel():
        s = select(pos_columns).where(portfolio_pos_t.c.ra_portfolio_id == globalid)
        pos_df = pd.read_sql(s, db, index_col = ['ra_date', 'ra_fund_code'], parse_dates = ['ra_date'])
        pos_df = pos_df.unstack(1).fillna(0.0)
        dates = pos_df.index.ravel()
        if optstartdate is not None:
            pos_df = pos_df[pos_df.index >= optstartdate]
        if optenddate is not None:
            pos_df = pos_df[pos_df.index <= optenddate]
        print globalid, ' change position num ' ,len(pos_df.index)
