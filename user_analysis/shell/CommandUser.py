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
import multiprocessing
from multiprocessing import Manager

from datetime import datetime, timedelta
from dateutil.parser import parse
from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from util import xdict
from util.xdebug import dd
from db import database, trade


import traceback, code

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def user(ctx):

    '''user analysis
    '''
    pass



@user.command()
@click.pass_context
def ts_order(ctx):

    #engine = database.connection('trade')
    #Session = sessionmaker(bind=engine)
    #session = Session()
    #sql = session.query(trade.ts_order.ts_uid, trade.ts_order.ts_trade_type, trade.ts_order.ts_trade_status, trade.ts_order.ts_placed_date, trade.ts_order.ts_risk, trade.ts_order.ts_placed_percent, trade.ts_order.ts_placed_amount).filter(trade.ts_order.ts_trade_type.in_([3,4,5,6])).statement
    #ts_order = pd.read_sql(sql, session.bind, index_col = ['ts_uid'], parse_dates = ['ts_placed_date'])
    #ts_order.to_csv('tmp/ts_order.csv')
    #session.commit()
    #session.close()
    #for k, v in ts_order.groupby(level = [0]):
    #    v = v.reset_index().set_index(['ts_placed_date']).sort_index()
    #    print(k, v)

    ts_order = pd.read_csv('tmp/ts_order.csv', index_col = ['ts_uid'], parse_dates = ['ts_placed_date'])


@user.command()
@click.pass_context
def ts_holding_nav(ctx):

    #engine = database.connection('trade')
    #Session = sessionmaker(bind=engine)
    #session = Session()
    #sql = session.query(trade.ts_holding_nav.ts_uid, trade.ts_holding_nav.ts_portfolio_id, trade.ts_holding_nav.ts_date, trade.ts_holding_nav.ts_nav, trade.ts_holding_nav.ts_asset, trade.ts_holding_nav.ts_profit).statement
    #ts_holding_nav = pd.read_sql(sql, session.bind, index_col = ['ts_uid'], parse_dates = ['ts_date'])
    #ts_holding_nav.to_csv('tmp/ts_holding_nav.csv')
    #session.commit()
    #session.close()
    #for k, v in ts_holding_nav.groupby(level = [0]):
    #    v = v.reset_index().set_index(['ts_date']).sort_index()
    #    print(k, v)

    ts_holding_nav = pd.read_csv('tmp/ts_holding_nav.csv', index_col = ['ts_uid'], parse_dates = ['ts_date'])
    print(ts_holding_nav)
