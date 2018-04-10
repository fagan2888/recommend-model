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


class tq_fd_basicinfo(Base):

    __tablename__ = 'tq_fd_basicinfo'

    ID = Column(Integer, primary_key = True)

    secode = Column(String)

    fdsname = Column(String)
    fdname = Column(String)

    fdtype = Column(String)

    fdnature = Column(String)
    fsymbol = Column(String)

    fdstyle = Column(String)


class yinhe_type(Base):

    __tablename__ = 'yinhe_type'

    yt_fund_id = Column(Integer, primary_key = True)

    yt_fund_code = Column(String)
    yt_l1_type = Column(String)
    yt_l1_name = Column(String)
    yt_l2_type = Column(String)
    yt_l2_name = Column(String)
    yt_l3_type = Column(String)
    yt_l3_name = Column(String)

    yt_end_date = Column(Date)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class tq_fd_typeclass(Base):

    __tablename__ = 'tq_fd_typeclass'

    securityid = Column(String, primary_key = True)

    l1codes = Column(String)
    l1name = Column(String)
    l2codes = Column(String)
    l2name = Column(String)
    l3codes = Column(String)
    l3name = Column(String)

    begindate = Column(Date)
    enddate = Column(Date)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class tq_oa_securitymap(Base):

    __tablename__ = 'tq_oa_securitymap'

    ID = Column(Integer, primary_key = True)

    secode = Column(String)
    maptype = Column(String)

    mapcode = Column(String)
    mapname = Column(String)

    enddate = Column(String)


class tq_oa_stcode(Base):

    __tablename__ = 'tq_oa_stcode'

    ID = Column(Integer, primary_key = True)
    securityid = Column(String)
    secode = Column(String)



class tq_ix_basicinfo(Base):

    __tablename__ = 'tq_ix_basicinfo'

    ID = Column(Integer, primary_key = True)

    secode = Column(String)

    indexname = Column(String)

    indexsname = Column(String)

    symbol = Column(String)



class tq_ix_mweight(Base):

    __tablename__ = 'tq_ix_mweight'

    ID = Column(Integer, primary_key = True)

    tradedate = Column(Date)
    secode = Column(String)

    constituentsecode = Column(String)
    constituentcode = Column(String)
    weight = Column(String)






class tq_fd_skdetail(Base):

    __tablename__ = 'tq_fd_skdetail'

    ID = Column(Integer, primary_key = True)

