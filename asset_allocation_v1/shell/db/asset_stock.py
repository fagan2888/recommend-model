#coding=utf8


import sys
sys.path.append('shell')
from sqlalchemy import MetaData, Table, select, func, and_
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
import pandas as pd
import MySQLdb
import logging
import database
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dateutil.parser import parse
import time
import asset
import numpy as np

logger = logging.getLogger(__name__)

Base = declarative_base()

class ra_stock(Base):

    __tablename__ = 'ra_stock'

    globalid = Column(String, primary_key = True)
    sk_code = Column(String)
    sk_name = Column(String)
    sk_secode = Column(String)
    sk_listdate = Column(Date)
    sk_compcode = Column(String)
    sk_swlevel1code = Column(String)

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

class tq_sk_finindic(Base):

    __tablename__ = 'tq_sk_finindic'

    ID = Column(Integer, primary_key = True)

    symbol = Column(String)
    secode = Column(String)
    tradedate = Column(Date)

    pettm = Column(Float)
    pettmnpaaei = Column(Float)
    pb = Column(Float)

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


def load_ohlcavntt(globalid):

    secode = asset.StockAsset.secode_dict()[globalid]

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_qt_skdailyprice.tradedate, tq_qt_skdailyprice.topen, tq_qt_skdailyprice.thigh, tq_qt_skdailyprice.tlow, tq_qt_skdailyprice.tclose, tq_qt_skdailyprice.vol, tq_qt_skdailyprice.amount, tq_qt_skdailyprice.negotiablemv, tq_qt_skdailyprice.totmktcap, tq_qt_skdailyprice.turnrate).filter(tq_qt_skdailyprice.secode == str(secode)).statement

    df = pd.read_sql(sql, session.bind, index_col = ['tradedate'], parse_dates = ['tradedate'])
    session.commit()
    session.close()

    df.turnrate = df.turnrate / 100

    return df


def load_fdmt(globalid):

    df_epbp = load_epbp(globalid)
    df_holder_avgpct = load_holder_avgpct(globalid)
    df_roe_roa = load_roe_roa(globalid)
    df_ccdfg = load_ccdfg(globalid)

    df = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True,how='outer'), [df_epbp, df_holder_avgpct, df_roe_roa, df_ccdfg])
    df = df.fillna(method = 'pad', limit = 126)

    return df


def load_epbp(globalid):

    secode = asset.StockAsset.secode_dict()[globalid]

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_finindic.tradedate, tq_sk_finindic.pettm, tq_sk_finindic.pb).filter(tq_sk_finindic.secode == secode).statement
    df = pd.read_sql(sql, session.bind, index_col = ['tradedate'], parse_dates = ['tradedate'])

    session.commit()
    session.close()

    df = df.loc[~df.index.duplicated(keep = 'last')]
    df = df[df.index > '1990']
    df = df.fillna(np.nan)
    return df


def load_holder_avgpct(globalid):

    compcode = asset.StockAsset.compcode_dict()[globalid]

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_sk_shareholdernum.publishdate, tq_sk_shareholdernum.aholdproportionpacc).filter(tq_sk_shareholdernum.compcode == compcode).statement
    df = pd.read_sql(sql, session.bind, index_col = ['publishdate'], parse_dates = ['publishdate'])

    session.commit()
    session.close()


    df = df.loc[~df.index.duplicated(keep = 'last')]
    df = df[df.index > '1990']
    df = df.fillna(np.nan)
    return df


def load_roe_roa(globalid):

    compcode = asset.StockAsset.compcode_dict()[globalid]

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_prottmindic.publishdate, tq_fin_prottmindic.roedilutedcut, tq_fin_prottmindic.roa).filter(and_(tq_fin_prottmindic.reporttype == 3, tq_fin_prottmindic.compcode == compcode)).statement
    df = pd.read_sql(sql, session.bind, index_col = ['publishdate'], parse_dates = ['publishdate'])

    session.commit()
    session.close()

    df = df.loc[~df.index.duplicated(keep = 'last')]
    df = df[df.index > '1990']
    df = df.fillna(np.nan)
    return df


def load_ccdfg(globalid):

    compcode = asset.StockAsset.compcode_dict()[globalid]

    engine = database.connection('caihui')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(tq_fin_proindicdata.firstpublishdate, tq_fin_proindicdata.currentrt, tq_fin_proindicdata.cashrt, tq_fin_proindicdata.ltmliabtota, tq_fin_proindicdata.equtotliab, tq_fin_proindicdata.sgpmargin).filter(tq_fin_proindicdata.reporttype == 3).filter(tq_fin_proindicdata.compcode == compcode).statement
    df = pd.read_sql(sql, session.bind, index_col = ['firstpublishdate'], parse_dates = ['firstpublishdate'])

    session.commit()
    session.close()

    df = df.loc[~df.index.duplicated(keep = 'last')]
    df = df[df.index > '1990']
    df = df.fillna(np.nan)

    return df


if __name__ == '__main__':

    # df = load_ohlcavntt('SK.601318')
    # df = load_epbp(2010000857)
    # df = load_avgpct(2010000857)
    # df = load_holder_avgpct(2010000857)
    # df = load_roe_roa(2010000857)
    # df = load_ccdfg(2010001036)
    # df = load_fdmt(2010000938)
    # print df.head()
    print asset.StockAsset.all_stock_info()
    pass
