#coding=utf8

import sys
sys.path.append('shell')
from sqlalchemy import MetaData, Table, select, func, and_
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
import numpy as np
import pandas as pd
import logging
from . import database
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from ipdb import set_trace
import asset

logger = logging.getLogger(__name__)

Base = declarative_base()


class tq_ix_basicinfo(Base):

    __tablename__ = 'tq_ix_basicinfo'

    id = Column(Integer, primary_key = True)
    secode = Column(String)
    indexname = Column(String)
    symbol = Column(String)
    estclass = Column(String)


class index_factor(Base):

    __tablename__ = 'index_factor'

    index_id = Column(String, primary_key=True)
    if_type = Column(Integer)
    secode = Column(String)
    index_name = Column(String)


class tq_qt_index(Base):

    __tablename__ = 'tq_qt_index'

    id = Column(Integer, primary_key=True)
    secode = Column(String)
    tradedate = Column(String)
    tclose = Column(Float)


def load_caihui_index(secodes=None, start_date=None, end_date=None):

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_qt_index.tradedate, tq_qt_index.secode, tq_qt_index.tclose)
    if secodes is not None:
        sql = sql.filter(tq_qt_index.secode.in_(secodes))
    if start_date is not None:
        sql = sql.filter(tq_qt_index.tradedate >= start_date)
    if end_date is not None:
        sql = sql.filter(tq_qt_index.tradedate <= end_date)
    df_nav = pd.read_sql(sql.statement, session.bind, index_col=['tradedate', 'secode'], parse_dates=['tradedate'])
    session.commit()
    session.close()

    df_nav = df_nav.unstack()
    df_nav.columns = df_nav.columns.get_level_values(1)

    return df_nav


def load_type_index(estclass=None):

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_ix_basicinfo.secode, tq_ix_basicinfo.symbol, tq_ix_basicinfo.indexname)
    if estclass is not None:
        sql = sql.filter(tq_ix_basicinfo.estclass.in_(estclass))
    index_info = pd.read_sql(sql.statement, session.bind, index_col=['secode'])
    session.commit()
    session.close()

    return index_info


def load_tq_ix_basicinfo(secodes = None):

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_ix_basicinfo.secode, tq_ix_basicinfo.symbol, tq_ix_basicinfo.indexname)
    if secodes is not None:
        sql = sql.filter(tq_ix_basicinfo.secode.in_(secodes))
    index_info = pd.read_sql(sql.statement, session.bind, index_col = ['secode'])
    session.commit()
    session.close()

    return index_info


def load_ix_secode_by_symbol(symbols = None):

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_ix_basicinfo.secode, tq_ix_basicinfo.symbol)
    if symbols is not None:
        sql = sql.filter(tq_ix_basicinfo.symbol.in_(symbols))
    all_indexes = pd.read_sql(sql.statement, session.bind, index_col = ['secode'])
    session.commit()
    session.close()

    return all_indexes


def load_all_index_factor(if_type = None):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(index_factor.index_id, index_factor.secode, index_factor.index_name)
    if if_type is not None:
        sql = sql.filter(index_factor.if_type == if_type)
    else:
        sql = sql.filter(index_factor.if_type > 0.0)
    all_indexes = pd.read_sql(sql.statement, session.bind, index_col = ['index_id'])
    session.commit()
    session.close()

    return all_indexes




if __name__ == '__main__':

    pass
