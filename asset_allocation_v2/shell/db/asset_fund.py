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


class tq_fd_skdetail(Base):

    __tablename__ = 'tq_fd_skdetail'

    id = Column(Integer, primary_key = True)
    publishdate = Column(Date)
    enddate = Column(Date)
    secode = Column(String)
    skcode = Column(String)
    navrto = Column(Float)


class tq_fd_basicinfo(Base):

    __tablename__ = 'tq_fd_basicinfo'

    id = Column(Integer, primary_key = True)
    secode = Column(String)
    fsymbol = Column(String)


class tq_fd_typeclass(Base):

    __tablename__ = 'tq_fd_typeclass'

    id = Column(Integer, primary_key = True)
    securityid = Column(String)
    l1codes = Column(String)


class tq_fd_sharestat(Base):

    __tablename__ = 'tq_fd_sharestat'

    id = Column(Integer, primary_key = True)
    secode = Column(String)
    publishdate = Column(Date)
    holdernum = Column(Float)
    avgshare = Column(Float)


class tq_oa_securitymap(Base):

    __tablename__ = 'tq_oa_securitymap'

    id = Column(Integer, primary_key = True)
    secode = Column(String)
    mapcode = Column(String)
    maptype = Column(String)


def load_securitymap(maptype = 25):

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_oa_securitymap.secode, tq_oa_securitymap.maptype).filter(tq_oa_securitymap.maptype == maptype)
    df = pd.read_sql(sql.statement, session.bind, index_col = ['secode'])
    session.commit()
    session.close()

    return df


def load_fund_by_type(l1codes = 2001):

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_fd_typeclass.securityid, tq_fd_typeclass.l1codes).filter(tq_fd_typeclass.l1codes == l1codes)
    df = pd.read_sql(sql.statement, session.bind, index_col = ['securityid'])
    session.commit()
    session.close()

    return df


def load_fund_index(fund_codes = None):

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_oa_securitymap.secode, tq_oa_securitymap.mapcode).filter(tq_oa_securitymap.maptype == 25)
    if fund_codes is not None:
        sql = sql.filter(tq_oa_securitymap.secode.in_(fund_codes))
    df = pd.read_sql(sql.statement, session.bind, index_col = ['secode'])
    session.commit()
    session.close()

    return df


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


def load_fund_unit_nav_series(code, reindex=None, begin_date=None, end_date=None):

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(ra_fund_nav.ra_date, ra_fund_nav.ra_nav).filter(ra_fund_nav.ra_code == code)
    if begin_date is not None:
        sql = sql.filter(ra_fund_nav.ra_date >= begin_date.strftime('%Y%m%d'))
    if end_date is not None:
        sql = sql.filter(ra_fund_nav.ra_date <= end_date.strftime('%Y%m%d'))
    df = pd.read_sql(sql.statement, session.bind, index_col=['ra_date'], parse_dates=['ra_date'])
    if reindex is not None:
        df = df.reindex(reindex, method='pad')
    ser = df.ra_nav
    ser.index.name = 'date'
    session.commit()
    session.close()

    return ser


def load_fund_secode_dict():

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_fd_basicinfo)
    df = pd.read_sql(sql.statement, session.bind)
    session.commit()
    session.close()

    secode_dict = dict(zip(df.secode.values, df.fsymbol.values))
    return secode_dict


def load_all_fund_pos():

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_fd_skdetail)
    df = pd.read_sql(sql.statement, session.bind, index_col=['secode'], parse_dates=['publishdate', 'enddate'])
    session.commit()
    session.close()

    secode_dict = load_fund_secode_dict()
    # df = pd.read_csv('data/fund_pos.csv', index_col=['secode'], parse_dates=['publishdate', 'enddate'])
    df.index = df.index.astype('str')
    df = df.rename(index = secode_dict)
    df = df.loc[:, ['publishdate', 'skcode', 'navrto']]
    stock_secode_dict = asset.StockAsset.secode_dict()
    stock_secode_dict = {v:k for k,v in stock_secode_dict.items()}
    df['skcode'] = df['skcode'].astype('str')
    # df['skcode'] = df['skcode'].replace(stock_secode_dict)
    df['skcode'] = [stock_secode_dict.get(x, np.nan) for x in df['skcode'].values]
    df = df.dropna()

    return df


def load_all_fund_share():

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_fd_sharestat)
    df = pd.read_sql(sql.statement, session.bind, parse_dates=['publishdate'])
    session.commit()
    session.close()

    secode_dict = load_fund_secode_dict()
    df['share'] = df['holdernum']*df['avgshare']
    df = df[['publishdate', 'secode', 'share']]
    df = df.groupby(['publishdate', 'secode']).last()
    df = df.unstack().fillna(method = 'pad')
    df = df.rename(columns=secode_dict)
    df = df.fillna(0.0)
    df.columns = df.columns.get_level_values(1)

    return df



if __name__ == '__main__':

    # df = load_fund_nav_series('519983')
    df = load_all_fund_share()







