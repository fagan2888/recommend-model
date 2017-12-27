#coding=utf8


from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime
import pandas as pd
import logging
import database
from sqlalchemy.ext.declarative import declarative_base
from dateutil.parser import parse


logger = logging.getLogger(__name__)


Base = declarative_base()


class stock_factor(Base):

    __tablename__ = 'stock_factor'

    sf_id = Column(String, primary_key = True)
    sf_name = Column(String)
    sf_explain = Column(Text)
    sf_source = Column(Integer)
    sf_kind = Column(String)
    sf_formula = Column(String)
    sf_start_date = Column(Date)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class ra_stock(Base):

    __tablename__ = 'ra_stock'

    globalid = Column(String, primary_key = True)
    sk_code = Column(String)
    sk_name = Column(String)
    sk_secode = Column(String)


    updated_at = Column(DateTime)
    created_at = Column(DateTime)




