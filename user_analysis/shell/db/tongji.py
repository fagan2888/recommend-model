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

class log_raw_apps(Base):

    __tablename__ = 'log_raw_apps'

    id = Column(Integer, primary_key = True)
    lr_date = Column(Date)
    lr_time = Column(DateTime)
    lr_uid = Column(Integer)
    lr_pid = Column(Integer)
    lr_page = Column(Integer)
    lr_ctrl = Column(Integer)
    lr_oid = Column(Integer)
    lr_tag = Column(Integer)
    lr_ref = Column(Integer)
    lr_ver = Column(Integer)
    lr_chn = Column(Integer)
    lr_os = Column(Integer)
    lr_flag = Column(Integer)
    lr_ev = Column(Integer)
    lr_ts = Column(Integer)
    lr_ip = Column(Integer)


    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class buried_versions(Base):

    __tablename__ = 'buried_versions'

    globalid = Column(Integer, primary_key = True)
    by_version = Column(String)
