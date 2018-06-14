#coding=utf8


from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
import pandas as pd
import logging
from . import database
from sqlalchemy.ext.declarative import declarative_base
from dateutil.parser import parse


logger = logging.getLogger(__name__)

Base = declarative_base()

class ra_stock(Base):

    __tablename__ = 'ra_stock'

    globalid = Column(String, primary_key = True)
    sk_code = Column(String)
    sk_name = Column(String)
    sk_secode = Column(String)
    sk_listdate = Column(Date)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class tq_sk_specialtrade(Base):


    __tablename__ = 'tq_sk_specialtrade'


    id = Column(Integer, primary_key = True)
    secode = Column(String)
    selectedtype = Column(Integer)
    selecteddate = Column(String)
    outdate = Column(String)


class tq_sk_dquoteindic(Base):

    __tablename__ = 'tq_sk_dquoteindic'

    id = Column(Integer, primary_key = True)
    secode = Column(String)
    symbol = Column(String)
    tradedate = Column(Date)
    topenaf  = Column(Float)
    thighaf  = Column(Float)
    tlowaf   = Column(Float)
    tcloseaf = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    mktshare = Column(Float)
    totalshare = Column(Float)


class tq_qt_skdailyprice(Base):

    __tablename__ = 'tq_qt_skdailyprice'

    id = Column(Integer, primary_key = True)
    secode = Column(String)
    symbol = Column(String)
    tradedate = Column(Date)
    topen  = Column(Float)
    thigh  = Column(Float)
    tlow   = Column(Float)
    tclose = Column(Float)
    pchg = Column(Float)
    amplitude = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    totmktcap = Column(Float)
    turnrate = Column(Float)


class tq_sk_yieldindic(Base):

    __tablename__ = 'tq_sk_yieldindic'

    id = Column(Integer, primary_key = True)
    secode = Column(String)
    symbol = Column(String)
    tradedate = Column(Date)

    Yield  = Column(Float)
    yieldw  = Column(Float)
    yieldm  = Column(Float)
    yield3m  = Column(Float)
    yield6m  = Column(Float)

    turnrate  = Column(Float)
    turnratem  = Column(Float)
    turnrate3m  = Column(Float)
    turnrate6m  = Column(Float)
    turnratey  = Column(Float)

