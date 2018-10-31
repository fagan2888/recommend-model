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

class ts_order(Base):

    __tablename__ = 'ts_order'

    id = Column(Integer, primary_key = True)
    ts_uid = Column(String)
    ts_portfolio_id = Column(String)
    ts_txn_id = Column(String)
    ts_portfolio_id = Column(String)
    ts_trade_type = Column(Integer)
    ts_trade_status = Column(Integer)
    ts_trade_date = Column(Date)
    ts_placed_date = Column(Date)
    ts_placed_time = Column(DateTime)
    ts_placed_amount = Column(Float)
    ts_placed_percent = Column(Float)
    ts_risk = Column(Float)


    updated_at = Column(DateTime)
    created_at = Column(DateTime)



class ts_order_fund(Base):

    __tablename__ = 'ts_order_fund'

    id = Column(Integer, primary_key = True)
    ts_uid = Column(String)
    ts_portfolio_id = Column(String)
    ts_txn_id = Column(String)
    ts_portfolio_id = Column(String)
    ts_trade_type = Column(Integer)
    ts_trade_status = Column(Integer)
    ts_trade_date = Column(Date)
    ts_placed_date = Column(Date)
    ts_placed_time = Column(DateTime)
    ts_placed_amount = Column(Float)
    ts_acked_amount = Column(Float)
    ts_risk = Column(Float)


    updated_at = Column(DateTime)
    created_at = Column(DateTime)




class ts_holding_nav(Base):

    __tablename__ = 'ts_holding_nav'

    ts_uid = Column(Integer, primary_key = True)
    ts_portfolio_id = Column(String, primary_key = True)
    ts_date = Column(Date)
    ts_nav = Column(Float)
    ts_inc = Column(Float)
    ts_share = Column(Float)
    ts_asset = Column(Float)
    ts_profit = Column(Float)
    ts_processing_asset = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)

class yingmi_accounts(Base):

    __tablename__ = 'yingmi_accounts'

    ya_uid = Column(Integer, primary_key = True)
    ya_name = Column(String)
