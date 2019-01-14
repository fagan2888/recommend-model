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
from db import database, trade, recommend, tongji, mapi
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pickle
from tempfile import NamedTemporaryFile
import functools
from ipdb import set_trace
import warnings
from esdata import ESData

warnings.filterwarnings("ignore")


import traceback, code

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def feature(ctx):

    '''user feature analysis
    '''



@feature.command()
@click.pass_context
def user_under_sample_feature(ctx):


    engine = database.connection('trade')
    Session = sessionmaker(bind=engine)
    session = Session()
    uids = session.query(distinct(trade.ts_order.ts_uid)).all()
    session.commit()
    session.close()


    def user_app_log(uid):

        uid = uid[0]
        query = {"query":{"match":{"uid":uid}}}
        logs = ESData().load_access_data(query)
        print(logs[-1])
        return

    def user_ts_order(uid):
        return

    def user_ts_holding(uid):
        return

    def user_feature(uid):

        user_app_log(uid)

        return 


    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[8]')
            .set("spark.executor.memory", "24G")
            .set('spark.driver.memory', '24G')
            .set('spark.driver.maxResultSize', '24G'))
    sc = SparkContext(conf=spark_conf)
    uid_rdd = sc.parallelize(uids)

    uid_rdd.map(user_feature).collect()


    return

