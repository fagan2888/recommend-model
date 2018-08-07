#coding=utf8

import sys
sys.path.append('shell')
from datetime import datetime
import pandas as pd
from ipdb import set_trace
from sqlalchemy import Column, String, Integer, Text, Date, DateTime, Float
from sqlalchemy import MetaData, Table, select, func, and_
from sqlalchemy.orm  import sessionmaker
import logging
from sqlalchemy.ext.declarative import declarative_base
from . import database
from sqlalchemy.sql.expression import func
import numpy as np
import time
from ipdb import set_trace


logger = logging.getLogger(__name__)

Base = declarative_base()

class fund_factor(Base):


    __tablename__ = 'fund_factor'

    ff_id = Column(String, primary_key = True)
    ff_name = Column(String)
    created_at = Column(DateTime)


class fund_factor_exposure(Base):

    __tablename__ = 'fund_factor_exposure'

    fund_id = Column(String, primary_key = True)
    ff_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    exposure = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class fund_factor_return(Base):

    __tablename__ = 'fund_factor_return'

    ff_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    ret = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class fund_factor_specific_return(Base):

    __tablename__ = 'fund_factor_specific_return'

    fund_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    sret = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


def load_fund_factor_exposure(fund_id = None, fund_ids = None, ff_id = None, ff_ids = None, begin_date = None, end_date = None):

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    record = session.query(
        fund_factor_exposure.fund_id,
        fund_factor_exposure.ff_id,
        fund_factor_exposure.trade_date,
        fund_factor_exposure.exposure,
        )

    if fund_id:
        record = record.filter(fund_factor_exposure.fund_id == fund_id)
    if fund_ids is not None:
        record = record.filter(fund_factor_exposure.fund_id.in_(fund_ids))
    if ff_id:
        record = record.filter(fund_factor_exposure.ff_id == ff_id)
    if ff_ids is not None:
        record = record.filter(fund_factor_exposure.ff_id.in_(ff_ids))
    if begin_date:
        record = record.filter(fund_factor_exposure.trade_date >= begin_date)
    if end_date:
        record = record.filter(fund_factor_exposure.trade_date <= end_date)

    df = pd.read_sql(record.statement, session.bind, index_col = ['fund_id', 'ff_id', 'trade_date'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    return df


def load_fund_factor_return(ff_id = None, ff_ids = None, trade_date = None, begin_date = None, end_date = None, reindex = None):

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    record = session.query(
        fund_factor_return.ff_id,
        fund_factor_return.trade_date,
        fund_factor_return.ret,
        )

    if ff_id:
        record = record.filter(fund_factor_return.ff_id == ff_id)
    if ff_ids is not None:
        record = record.filter(fund_factor_return.ff_id.in_(ff_ids))
    if trade_date:
        record = record.filter(fund_factor_return.trade_date == trade_date)
    if begin_date:
        record = record.filter(fund_factor_return.trade_date >= begin_date)
    if end_date:
        record = record.filter(fund_factor_return.trade_date <= end_date)

    df = pd.read_sql(record.statement, session.bind, index_col = ['ff_id', 'trade_date'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    return df


def load_fund_factor_specific_return(fund_id = None, fund_ids = None, trade_date = None, begin_date = None, end_date = None):

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    record = session.query(
        fund_factor_specific_return.fund_id,
        fund_factor_specific_return.trade_date,
        fund_factor_specific_return.sret,
        )

    if fund_id:
        record = record.filter(fund_factor_specific_return.fund_id == fund_id)
    if fund_ids is not None:
        record = record.filter(fund_factor_specific_return.fund_id.in_(fund_ids))
    if trade_date:
        record = record.filter(fund_factor_specific_return.trade_date == trade_date)
    if begin_date:
        record = record.filter(fund_factor_specific_return.trade_date >= begin_date)
    if end_date:
        record = record.filter(fund_factor_specific_return.trade_date <= end_date)

    df = pd.read_sql(record.statement, session.bind, index_col = ['fund_id', 'trade_date'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    return df


def update_exposure(ff, last_date = None):

    ff_id = ff.factor_id
    exposure = ff.exposure

    if last_date is None:

        db = database.connection('asset')
        Session = sessionmaker(bind = db)
        session = Session()
        record = session.query(func.max(fund_factor_exposure.trade_date)).filter(fund_factor_exposure.ff_id == ff_id).first()
        last_date = record[0].strftime('%Y-%m-%d') if record[0] is not None else '1900-01-01'
        session.commit()
        session.close()

    exposure = exposure[exposure.index >= last_date]

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    session.query(fund_factor_exposure).filter(fund_factor_exposure.trade_date >= last_date).filter(fund_factor_exposure.ff_id == ff_id).delete()
    session.commit()
    session.close()

    df_new = exposure.stack()
    df_new = df_new.reset_index()
    df_new['ff_id'] = ff.factor_id
    df_new.columns = ['trade_date', 'fund_id', 'exposure', 'ff_id']
    df_new = df_new.set_index(['fund_id', 'ff_id', 'trade_date'])

    db = database.connection('asset')
    t = Table('fund_factor_exposure', MetaData(bind=db), autoload = True)
    database.batch(db, t, df_new, pd.DataFrame())


def update_fund_factor_return(df_ret, last_date = None):

    if last_date is None:

        db = database.connection('asset')
        Session = sessionmaker(bind = db)
        session = Session()
        record = session.query(func.max(fund_factor_return.trade_date)).first()
        last_date = record[0].strftime('%Y-%m-%d') if record[0] is not None else '1900-01-01'
        session.commit()
        session.close()

    df_ret = df_ret[df_ret.index >= last_date]

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    session.query(fund_factor_return).filter(fund_factor_return.trade_date >= last_date).delete()
    session.commit()
    session.close()

    df_ret = df_ret.stack()
    df_ret = df_ret.reset_index()
    df_ret.columns = ['trade_date', 'ff_id', 'ret']
    df_ret = df_ret.set_index(['ff_id', 'trade_date'])

    db = database.connection('asset')
    t = Table('fund_factor_return', MetaData(bind=db), autoload = True)
    database.batch(db, t, df_ret, pd.DataFrame())


def update_fund_factor_specific_return(df_sret, last_date = None):

    if last_date is None:

        db = database.connection('asset')
        Session = sessionmaker(bind = db)
        session = Session()
        record = session.query(func.max(fund_factor_specific_return.trade_date)).first()
        last_date = record[0].strftime('%Y-%m-%d') if record[0] is not None else '1900-01-01'
        session.commit()
        session.close()

    df_sret = df_sret[df_sret.index >= last_date]

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    session.query(fund_factor_specific_return).filter(fund_factor_specific_return.trade_date >= last_date).delete()
    session.commit()
    session.close()

    df_sret = df_sret.stack()
    df_sret = df_sret.reset_index()
    df_sret.columns = ['trade_date', 'fund_id', 'sret']
    df_sret = df_sret.set_index(['fund_id', 'trade_date'])

    db = database.connection('asset')
    t = Table('fund_factor_specific_return', MetaData(bind=db), autoload = True)
    database.batch(db, t, df_sret, pd.DataFrame())


if __name__ == '__main__':

    #df1 = load_fund_factor_return()
    #df2 = load_fund_factor_specific_return()
    #set_trace()
    #update_exposure(fundFactor.SizefundFactor(factor_id = 'ff.000001'))
    set_trace()









