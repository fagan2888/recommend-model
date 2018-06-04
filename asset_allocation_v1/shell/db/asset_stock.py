#coding=utf8


from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
import pandas as pd
import logging
import database
from sqlalchemy.orm import sessionmaker
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
    negotiablemv = Column(Float)


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




def load_stock_nav_series(asset_id, reindex=None, begin_date=None, end_date=None):

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    record = session.query(ra_stock.sk_secode).filter(ra_stock.globalid == asset_id).first()
    session.commit()
    session.close()
    secode = record[0]

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(tq_sk_dquoteindic.tradedate ,tq_sk_dquoteindic.tcloseaf).filter(tq_sk_dquoteindic.secode == secode)
    if begin_date is not None:
        sql = sql.filter(tq_sk_dquoteindic.tradedate >= begin_date.strftime('%Y%m%d'))
    if end_date is not None:
        sql = sql.filter(tq_sk_dquoteindic.tradedate <= end_date.strftime('%Y%m%d'))
    df = pd.read_sql(sql.statement, session.bind, index_col=['tradedate'], parse_dates=['tradedate'])
    if reindex is not None:
        df = df.reindex(reindex, method='pad')
    ser = df.tcloseaf
    ser.index.name = 'date'
    session.commit()
    session.close()

    return ser


def globalid_2_name(globalid):

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    record = session.query(ra_stock.sk_name).filter(ra_stock.globalid == globalid).first()
    session.commit()
    session.close()
    name = record[0]
    return name


def load_ohlcavntt(globalid):


    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    record = session.query(ra_stock.sk_secode).filter(ra_stock.globalid == globalid).first()
    session.commit()
    session.close()
    secode = record[0]


    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.topen, tq_qt_skdailyprice.thigh, tq_qt_skdailyprice.tlow, tq_qt_skdailyprice.tclose, tq_qt_skdailyprice.vol, tq_qt_skdailyprice.amount, tq_qt_skdailyprice.negotiablemv, tq_qt_skdailyprice.totmktcap, tq_qt_skdailyprice.turnrate).filter(tq_qt_skdailyprice.secode == secode).statement


    df = pd.read_sql(sql, session.bind, index_col = ['tradedate'], parse_dates = ['tradedate'])
    session.commit()
    session.close()

    df.turnrate = df.turnrate / 100

    return df
