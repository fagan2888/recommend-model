#coding=utf8


import sys
sys.path.append('shell')
from sqlalchemy import MetaData, Table, select, func, and_
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
import pandas as pd
import MySQLdb
import logging
from . import database
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dateutil.parser import parse
import time
import numpy as np
from functools import reduce
from ipdb import set_trace

logger = logging.getLogger(__name__)

Base = declarative_base()


class user_questionnaire_answers(Base):

    __tablename__ = 'user_questionnaire_answers'

    id = Column(Integer, primary_key = True)
    uq_questionnaire_id = Column(Integer)
    uq_uid = Column(Integer)
    uq_question_id = Column(Integer)
    uq_answer = Column(String)
    uq_question_type = Column()
    uq_start_time = Column(DateTime)
    uq_end_time = Column(DateTime)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class user_risk_analyze_results(Base):

    __tablename__ = 'user_risk_analyze_results'

    id = Column(Integer, primary_key = True)
    ur_uid = Column(String)
    ur_nare_id = Column(Integer)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class user_questionnaire_summaries(Base):

    __tablename__ = 'user_questionnaire_summaries'

    id = Column(Integer, primary_key = True)
    uq_uid = Column(Integer)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)
