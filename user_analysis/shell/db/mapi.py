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

class wechat_users(Base):

    __tablename__ = 'wechat_users'

    id = Column(Integer, primary_key = True)
    uid = Column(Integer)
    mobile = Column(String)
    service_id = Column(Integer)


    updated_at = Column(DateTime)
    created_at = Column(DateTime)

