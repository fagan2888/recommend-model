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
from db import database, trade, recommend, tongji, mapi, base_ra_index_nav, base_trade_dates
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pickle
from tempfile import NamedTemporaryFile
import functools
from ipdb import set_trace
from esdata import ESData
import json
import warnings
warnings.filterwarnings("ignore")

import traceback, code

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def stream(ctx):

    '''user analysis
    '''
    pass


@stream.command()
@click.pass_context
def app_log_update(ctx):

    dates = pd.date_range('2017-01-01', datetime.now()).strftime('%Y-%m-%d').tolist()
    dates.sort()
    dirs = os.listdir('data/app_log')
    exited_dates = []
    for f in dirs:
        exited_dates.append(f[8:18])
    exited_dates.sort()
    exited_dates = exited_dates[0:-30]
    new_dates = list(set(dates).difference(set(exited_dates)))
    new_dates.sort()
    print(new_dates)

    def app_log_date_update(new_date):
        start_timestamp = time.mktime(time.strptime(new_date, '%Y-%m-%d')) * 1000
        end_timestamp = start_timestamp + 24 * 60 * 60 *1000
        query = {"query":{"range":{"c_time":{
                            "lt":int(end_timestamp),
                            "gt":int(start_timestamp)
                                    }
                            }
                        }
                }
        logs = ESData().load_access_data(query)
        f = open('data/app_log/app_log_%s' % new_date, 'w')
        for log in logs:
            f.write(str(log))
            f.write('\n')
        f.close()
        return None


    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[8]')
            .set("spark.executor.memory", "8G")
            .set('spark.driver.memory', '8G')
            .set('spark.driver.maxResultSize', '8G'))
    sc = SparkContext(conf=spark_conf)
    new_dates_rdd = sc.parallelize(new_dates)

    new_dates_rdd.map(app_log_date_update).collect()

    pass



@stream.command()
@click.pass_context
def ts_order_stream(ctx):

    engine = database.connection('trade')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(trade.ts_order.ts_uid, trade.ts_order.ts_txn_id, trade.ts_order.ts_portfolio_id ,trade.ts_order.ts_trade_type, trade.ts_order.ts_trade_status, trade.ts_order.ts_trade_date, trade.ts_order.ts_risk, trade.ts_order.ts_placed_percent, trade.ts_order.ts_placed_amount, trade.ts_order.ts_placed_date, trade.ts_order.ts_placed_time, trade.ts_order.ts_acked_date, trade.ts_order.ts_acked_amount, trade.ts_order.ts_acked_fee).filter(trade.ts_order.ts_trade_type.in_([3,4,5,6])).statement
    ts_order_df = pd.read_sql(sql, session.bind)
    session.commit()
    session.close()

    from pymongo import MongoClient
    conn = MongoClient('127.0.0.1', 27017)
    db = conn.user_analysis
    ts_order_collection = db.ts_order


    for record in ts_order_df.to_dict('records'):
        ts_placed_date = record['ts_placed_date']
        ts_placed_time = record['ts_placed_time']
        if ts_placed_date is None:
            continue
        ts_placed_datetime = datetime(ts_placed_date.year, ts_placed_date.month, ts_placed_date.day) + ts_placed_time
        record['ts_placed_datetime'] = ts_placed_datetime
        print(record)
        ts_order_collection.insert(json.dumps(record))



@stream.command()
@click.pass_context
def ts_holding(ctx):

    engine = database.connection('trade')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(trade.ts_holding_nav.ts_uid, trade.ts_holding_nav.ts_portfolio_id, trade.ts_holding_nav.ts_date, trade.ts_holding_nav.ts_nav, trade.ts_holding_nav.ts_asset, trade.ts_holding_nav.ts_profit, trade.ts_holding_nav.ts_processing_asset).statement
    ts_holding_nav = pd.read_sql(sql, session.bind)
    ts_holding_nav = ts_holding_nav.rename(columns = {'ts_uid':'ts_holding_uid'})
    ts_holding_nav = ts_holding_nav.set_index(['ts_holding_uid'])
    session.commit()
    session.close()

    ts_holding_nav.to_csv('data/ts_holding_nav.csv')
