#coding=utf8


from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
import pandas as pd
import logging
import database
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dateutil.parser import parse


logger = logging.getLogger(__name__)

Base = declarative_base()

class ra_fund(Base):

    __tablename__ = 'ra_fund'

    globalid = Column(Integer, primary_key = True)
    ra_code = Column(String)
    ra_type = Column(String)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)

class ra_fund_nav(Base):

    __tablename__ = 'ra_fund_nav'

    ra_fund_id = Column(Integer, primary_key = True)
    ra_date = Column(Date)
    ra_nav = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)

