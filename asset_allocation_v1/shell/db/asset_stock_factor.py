#coding=utf8

import pandas as pd
from ipdb import set_trace
from sqlalchemy import Column, String, Integer, Text, Date, DateTime, Float
from sqlalchemy import MetaData, Table, select, func, and_
from sqlalchemy.orm  import sessionmaker
import logging
from sqlalchemy.ext.declarative import declarative_base
import database

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


class stock_factor_exposure(Base):

    __tablename__ = 'stock_factor_exposure'

    stock_id = Column(String, primary_key = True)
    sf_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    exposure = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class stock_factor_return(Base):

    __tablename__ = 'stock_factor_return'

    sf_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    ret = Column(Float)
    sret = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class stock_factor_value(Base):

    __tablename__ = 'stock_factor_value'

    stock_id = Column(String, primary_key = True)
    sf_id = Column(String, primary_key = True)
    trade_date = Column(String, primary_key = True)
    factor_value = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class valid_stock_factor(Base):

    __tablename__ = 'valid_stock_factor'

    stock_id = Column(String, primary_key = True)
    secode   = Column(String)
    trade_date = Column(String, primary_key = True)
    valid = Column(Integer)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


def load_stock_factor_exposure(stock_id = None, sf_id = None, trade_date = None):

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    record = session.query(
        stock_factor_exposure.stock_id,
        stock_factor_exposure.sf_id,
        stock_factor_exposure.trade_date,
        stock_factor_exposure.exposure,
        )

    if stock_id:
        record = record.filter(stock_factor_exposure.stock_id == stock_id)
    if sf_id:
        record = record.filter(stock_factor_exposure.sf_id == sf_id)
    if trade_date:
        record = record.filter(stock_factor_exposure.trade_date == trade_date)

    df = pd.read_sql(record.statement, session.bind, index_col = ['stock_id', 'sf_id', 'trade_date'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    return df


def load_stock_factor_return(sf_id = None, trade_date = None):

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    record = session.query(
        stock_factor_return.sf_id,
        stock_factor_return.trade_date,
        stock_factor_return.ret,
        stock_factor_return.sret,
        )

    if sf_id:
        record = record.filter(stock_factor_return.sf_id == sf_id)
    if trade_date:
        record = record.filter(stock_factor_return.trade_date == trade_date)

    df = pd.read_sql(record.statement, session.bind, index_col = ['sf_id', 'trade_date'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    return df




if __name__ == '__main__':

    df = load_stock_factor_return('SF.000001')
    set_trace()








