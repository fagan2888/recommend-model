#coding=utf8


from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
import pandas as pd
import logging
import database
from sqlalchemy.ext.declarative import declarative_base
from dateutil.parser import parse


logger = logging.getLogger(__name__)

Base = declarative_base()

class t_macro_msupply(Base):

    __tablename__ = 't_macro_msupply'

    ID = Column(Integer, primary_key = True)
    nyear = Column(Integer)
    nmonth = Column(Integer)
    value_m2 = Column(Float)
    growthrate_m2 = Column(Float)
    growthrate_m1 = Column(Float)
    entrydate = Column(Date)


class t_macro_rlestindex(Base):

    __tablename__ = 't_macro_rlestindex'

    ID = Column(Integer, primary_key = True)
    nyear = Column(Integer)
    nmonth = Column(Integer)
    index_gc = Column(Float)
    entrydate = Column(Date)

class tq_ix_finindex(Base):

    __tablename__ = 'tq_ix_finindex'

    ID =  Column(Integer, primary_key = True)
    secode = Column(Integer)
    publishdate = Column(Integer)
    epscut = Column(Float)

class t_macro_qgdp(Base):

    __tablename__ = 't_macro_qgdp'

    ID = Column(Integer, primary_key = True)
    nyear = Column(Integer)
    nmonth = Column(Integer)
    value = Column(Float)

class tq_qt_cbdindex(Base):

    __tablename__ = 'tq_qt_cbdindex'

    ID = Column(Integer, primary_key = True)
    tradedate = Column(Integer)
    secode = Column(Float)
    avgmktcapmatyield = Column(Float)
