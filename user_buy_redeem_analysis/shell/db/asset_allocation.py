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


class high_quality_inferior_user(Base):

    __tablename__ = 'high_quality_inferior_user'

    uid = Column(Integer, primary_key = True)
    score = Column(Float)
    service_score = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)

class user_account_infos(Base):

    __tablename__ = 'user_account_infos'

    ua_uid = Column(Integer, primary_key = True)
    ua_uq_score = Column(Float)
    ua_uqs_score = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)
