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
from db import database, trade, recommend
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pickle
from tempfile import NamedTemporaryFile
from ipdb import set_trace
import warnings
warnings.filterwarnings("ignore")


import traceback, code

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def question(ctx):

    '''user analysis
    '''
    pass



@question.command()
@click.pass_context
def user_question_answer(ctx):

    engine = database.connection('recommend')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(recommend.user_questionnaire_answers.uq_uid, recommend.user_questionnaire_answers.uq_questionnaire_id, recommend.user_questionnaire_answers.uq_question_id, recommend.user_questionnaire_answers.uq_answer, recommend.user_questionnaire_answers.uq_question_type, recommend.user_questionnaire_answers.uq_start_time, recommend.user_questionnaire_answers.uq_end_time).statement
    user_question_answer_df = pd.read_sql(sql, session.bind, index_col = ['uq_uid','uq_questionnaire_id'], parse_dates = ['uq_start_time', 'uq_end_time'])
    session.commit()
    session.close()
    user_question_answer_df.to_csv('tmp/user_question_answer.csv')


    spark_conf = (SparkConf().setAppName('order holding').setMaster('local[16]')
            .set("spark.executor.memory", "50G")
            .set('spark.driver.memory', '50G')
            .set('spark.driver.maxResultSize', '50G')
            .set('spark.cores.max', 32))

    spark = SparkSession.builder.config(conf = spark_conf).getOrCreate()
    df = spark.read.csv("tmp/user_question_answer.csv", header = True)
    user_question_answer_rdd = df.rdd.repartition(1000)

    def question_answer_feature(v):
        try:
            k = v[0]
            vs = v[1]
            datas = [v.asDict() for v in vs]
            v = pd.DataFrame(datas).dropna()
            v.uq_questionnaire_id = v.uq_questionnaire_id.astype(int)
            v.uq_question_type = v.uq_question_type.astype(int)
            v = v[v.uq_questionnaire_id == max(v.uq_questionnaire_id)]
            v = v[v.uq_question_type == 0]
            v.uq_question_id = v.uq_question_id.astype(str)
            v['uq_question_answer'] = v.uq_question_id + v.uq_answer
            v = v.drop(columns = ['uq_question_type', 'uq_start_time', 'uq_end_time' ,'uq_question_id', 'uq_answer', 'uq_questionnaire_id'])
            v = v.set_index(['uq_uid', 'uq_question_answer'])
            v['tag'] = 1.0
            v = v[~v.index.duplicated()]
            v = v.unstack().fillna(0.0)
            v.columns = v.columns.droplevel(0)
        except:
            return pd.DataFrame()
        return v


    vs = user_question_answer_rdd.groupBy(lambda row : row.uq_uid).map(question_answer_feature).collect()
    user_question_answer_df = pd.concat(vs, axis = 0).fillna(0.0)
    print(user_question_answer_df.tail())
    user_question_answer_df.to_csv('tmp/user_question_answer_feature.csv')
