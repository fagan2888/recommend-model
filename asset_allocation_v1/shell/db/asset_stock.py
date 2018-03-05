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

class ra_stock(Base):

    __tablename__ = 'ra_stock'

    globalid = Column(String, primary_key = True)
    sk_code = Column(String)
    sk_name = Column(String)
    sk_secode = Column(String)
    sk_compcode = Column(String)
    sk_listdate = Column(Date)
    sk_swlevel1code = Column(String)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class tq_sk_specialtrade(Base):

    __tablename__ = 'tq_sk_specialtrade'

    ID = Column(Integer, primary_key = True)
    secode = Column(String)
    selectedtype = Column(Integer)
    selecteddate = Column(String)
    outdate = Column(String)


class tq_sk_dquoteindic(Base):

    __tablename__ = 'tq_sk_dquoteindic'

    ID = Column(Integer, primary_key = True)
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

    ID = Column(Integer, primary_key = True)
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

    ID = Column(Integer, primary_key = True)
    secode = Column(String)
    symbol = Column(String)
    tradedate = Column(Date)

    Yield  = Column(Float)
    yieldw  = Column(Float)
    yieldm  = Column(Float)
    yield3m  = Column(Float)
    yield6m  = Column(Float)
    yieldy  = Column(Float)

    turnrate  = Column(Float)
    turnratem  = Column(Float)
    turnrate3m  = Column(Float)
    turnrate6m  = Column(Float)
    turnratey  = Column(Float)


class tq_fin_proindicdata(Base):

    __tablename__ = 'tq_fin_proindicdata'

    ID = Column(Integer, primary_key = True)

    compcode = Column(String)
    firstpublishdate = Column(Date)
    enddate = Column(Date)
    naps = Column(Float)
    taturnrt = Column(Float)
    currentrt = Column(Float)
    cashrt = Column(Float)
    roa = Column(Float)
    roediluted = Column(Float)
    sgpmargin = Column(Float)
    npcut = Column(Float)
    fcff = Column(Float)
    fcfe = Column(Float)
    ltmliabtoequconms = Column(Float)
    ltmliabtota = Column(Float)
    equtotliab = Column(Float)
    reporttype = Column(Integer)

class tq_fin_proindicdatasub(Base):

    __tablename__ = 'tq_fin_proindicdatasub'

    ID = Column(Integer, primary_key = True)

    compcode = Column(String)
    firstpublishdate = Column(Date)
    enddate = Column(Date)
    grossprofit = Column(Float)
    reporttype = Column(Integer)


class tq_sk_finindic(Base):

    __tablename__ = 'tq_sk_finindic'

    ID = Column(Integer, primary_key = True)

    symbol = Column(String)
    secode = Column(String)
    tradedate = Column(Date)

    pettm = Column(Float)
    pettmnpaaei = Column(Float)
    pb = Column(Float)


class tq_sk_sharestruchg(Base):

    __tablename__ = 'tq_sk_sharestruchg'

    ID = Column(Integer, primary_key = True)

    compcode = Column(String)
    publishdate = Column(Date)
    begindate = Column(Date)
    enddate = Column(Date)

    fcircaamt = Column(Float)


class tq_sk_shareholdernum(Base):

    __tablename__ = 'tq_sk_shareholdernum'

    ID = Column(Integer, primary_key = True)

    compcode = Column(String)
    publishdate = Column(Date)
    updatedate = Column(Date)
    enddate = Column(Date)

    totalshamt = Column(Float)
    totalshrto = Column(Float)
    totalshare = Column(Float)

    askshamt = Column(Float)

    aholdproportionpacc = Column(Float)
    aproportiongrq = Column(Float)
    aproportiongrhalfyear = Column(Float)



class tq_fin_prottmindic(Base):

    __tablename__ = 'tq_fin_prottmindic'

    ID = Column(Integer, primary_key = True)

    compcode = Column(String)
    publishdate = Column(Date)
    updatedate = Column(Date)
    enddate = Column(Date)
    reporttype = Column(Integer)

    roedilutedcut = Column(Float)
    roa = Column(Float)


class tq_sk_businfo(Base):

    __tablename__ = 'tq_sk_businfo'

    ID = Column(Integer, primary_key = True)

    compcode = Column(String)
    publishdate = Column(Date)

    clgpmrto = Column(Float)
    corebizcostrto = Column(Float)



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
