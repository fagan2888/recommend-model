#coding=utf8

import sys
sys.path.append('shell')
from sqlalchemy import MetaData, Table, select, func, and_
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
import pandas as pd
import logging
from . import database
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from ipdb import set_trace

logger = logging.getLogger(__name__)

Base = declarative_base()

class ra_fund(Base):

    __tablename__ = 'ra_fund'

    globalid = Column(Integer, primary_key = True)
    ra_code = Column(String)
    ra_name = Column(String)
    ra_type = Column(Integer)
    ra_fund_type = Column(Integer)
    ra_mask = Column(Integer)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class ra_fund_nav(Base):

    __tablename__ = 'ra_fund_nav'

    ra_fund_id = Column(Integer, primary_key = True)
    ra_code = Column(String)
    ra_date = Column(Date)
    ra_type = Column(Integer)
    ra_nav = Column(Float)
    ra_inc = Column(Float)
    ra_nav_acc = Column(Float)
    ra_nav_adjusted = Column(Float)
    ra_inc_adjusted = Column(Float)
    ra_return_daily = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


def load_fund_nav_series(code, reindex=None, begin_date=None, end_date=None):

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(ra_fund_nav.ra_date, ra_fund_nav.ra_nav_adjusted).filter(ra_fund_nav.ra_code == code)
    if begin_date is not None:
        sql = sql.filter(ra_fund_nav.ra_date >= begin_date.strftime('%Y%m%d'))
    if end_date is not None:
        sql = sql.filter(ra_fund_nav.ra_date <= end_date.strftime('%Y%m%d'))
    df = pd.read_sql(sql.statement, session.bind, index_col=['ra_date'], parse_dates=['ra_date'])
    if reindex is not None:
        df = df.reindex(reindex, method='pad')
    ser = df.ra_nav_adjusted
    ser.index.name = 'date'
    session.commit()
    session.close()

    return ser




if __name__ == '__main__':

    df = load_fund_nav_series('519983')
    set_trace()
