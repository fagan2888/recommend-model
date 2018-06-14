#coding=utf8

import sys
sys.path.append('shell')
import pandas as pd
from ipdb import set_trace
from sqlalchemy import Column, String, Integer, Text, Date, DateTime, Float
from sqlalchemy import MetaData, Table, select, func, and_
from sqlalchemy.orm  import sessionmaker
import logging
from sqlalchemy.ext.declarative import declarative_base
import database
from sqlalchemy.sql.expression import func

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

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class stock_factor_specific_return(Base):

    __tablename__ = 'stock_factor_specific_return'

    stock_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    sret = Column(Float)

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


def load_stock_factor_exposure(stock_id = None, stock_ids = None, sf_id = None, begin_date = None, end_date = None):

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
    if stock_ids is not None:
        record = record.filter(stock_factor_exposure.stock_id.in_(stock_ids))
    if sf_id:
        record = record.filter(stock_factor_exposure.sf_id == sf_id)
    if begin_date:
        record = record.filter(stock_factor_exposure.trade_date >= begin_date)
    if end_date:
        record = record.filter(stock_factor_exposure.trade_date <= end_date)

    df = pd.read_sql(record.statement, session.bind, index_col = ['stock_id', 'sf_id', 'trade_date'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    return df


def load_stock_factor_return(sf_id = None, trade_date = None, begin_date = None, end_date = None, reindex = None):

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    record = session.query(
        stock_factor_return.sf_id,
        stock_factor_return.trade_date,
        stock_factor_return.ret,
        )

    if sf_id:
        record = record.filter(stock_factor_return.sf_id == sf_id)
    if trade_date:
        record = record.filter(stock_factor_return.trade_date == trade_date)
    if begin_date:
        record = record.filter(stock_factor_return.trade_date >= begin_date)
    if end_date:
        record = record.filter(stock_factor_return.trade_date <= end_date)

    df = pd.read_sql(record.statement, session.bind, index_col = ['sf_id', 'trade_date'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    return df


def load_stock_factor_specific_return(stock_id = None, stock_ids = None, trade_date = None, begin_date = None, end_date = None):

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    record = session.query(
        stock_factor_specific_return.stock_id,
        stock_factor_specific_return.trade_date,
        stock_factor_specific_return.sret,
        )

    if stock_id:
        record = record.filter(stock_factor_specific_return.stock_id == stock_id)
    if stock_ids is not None:
        record = record.filter(stock_factor_specific_return.stock_id.in_(stock_ids))
    if trade_date:
        record = record.filter(stock_factor_specific_return.trade_date == trade_date)
    if begin_date:
        record = record.filter(stock_factor_specific_return.trade_date >= begin_date)
    if end_date:
        record = record.filter(stock_factor_specific_return.trade_date <= end_date)

    df = pd.read_sql(record.statement, session.bind, index_col = ['stock_id', 'trade_date'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    return df


def update_exposure(sf, last_date = None):

    sf_id = sf.factor_id
    exposure = sf.exposure

    if last_date is None:

        db = database.connection('asset')
        Session = sessionmaker(bind = db)
        session = Session()
        record = session.query(func.max(stock_factor_exposure.trade_date)).filter(stock_factor_exposure.sf_id == sf_id).first()
        if record[0] is None:
            last_date = '1900-01-01'
        else:
            last_date = record[0]
            last_date = last_date.strftime('%Y-%m-%d')
        session.commit()
        session.close()

    exposure = exposure[exposure.index >= last_date]

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    session.query(stock_factor_exposure).filter(stock_factor_exposure.trade_date >= last_date).filter(stock_factor_exposure.sf_id == sf_id).delete()
    session.commit()
    session.close()

    df_new = exposure.stack()
    df_new = df_new.reset_index()
    df_new['sf_id'] = sf.factor_id
    df_new.columns = ['trade_date', 'stock_id', 'exposure', 'sf_id']
    df_new = df_new.set_index(['stock_id', 'sf_id', 'trade_date'])

    db = database.connection('asset')
    t = Table('stock_factor_exposure', MetaData(bind=db), autoload = True)
    database.batch(db, t, df_new, pd.DataFrame(index = df_new.index, columns = df_new.columns))



if __name__ == '__main__':

    #df1 = load_stock_factor_return()
    #df2 = load_stock_factor_specific_return()
    #set_trace()
    pass
    #update_exposure(StockFactor.SizeStockFactor(factor_id = 'SF.000001'))








