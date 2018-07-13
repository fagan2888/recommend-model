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
import database
from sqlalchemy.sql.expression import func
import numpy as np
import time
from ipdb import set_trace

from mongo import MyMongoDB
import config_mongo

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
    trade_date = Column(String, primary_key = True)
    valid = Column(Integer)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


def load_stock_factor_exposure(stock_id = None, stock_ids = None, sf_id = None, sf_ids = None, begin_date = None, end_date = None):

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
    if sf_ids is not None:
        record = record.filter(stock_factor_exposure.sf_id.in_(sf_ids))
    if begin_date:
        record = record.filter(stock_factor_exposure.trade_date >= begin_date)
    if end_date:
        record = record.filter(stock_factor_exposure.trade_date <= end_date)

    df = pd.read_sql(record.statement, session.bind, index_col = ['stock_id', 'sf_id', 'trade_date'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    return df


def load_stock_factor_return(sf_id = None, sf_ids = None, trade_date = None, begin_date = None, end_date = None, reindex = None):

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
    if sf_ids is not None:
        record = record.filter(stock_factor_return.sf_id.in_(sf_ids))
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
        last_date = record[0].strftime('%Y-%m-%d') if record[0] is not None else '1900-01-01'
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
    database.batch(db, t, df_new, pd.DataFrame())



def update_valid_stock_table(quotation):

	engine = database.connection('asset')
	Session = sessionmaker(bind=engine)
	session = Session()
	record = session.query(func.max(valid_stock_factor.trade_date)).first()
        last_date = record[0].strftime('%Y-%m-%d') if record[0] is not None else '1900-01-01'

	quotation = quotation[quotation.index >= last_date]

        session.query(valid_stock_factor).filter(valid_stock_factor.trade_date >= last_date).delete()


        for globalid in quotation.columns:
            records = []
            for date in quotation.index:
		value = quotation.loc[date, globalid]
		if np.isnan(value):
                    continue
                valid_stock = valid_stock_factor()
                valid_stock.stock_id = globalid
                valid_stock.trade_date = date
                valid_stock.valid = 1.0
                records.append(valid_stock)

            session.add_all(records)
            session.commit()

        logger.info('stock validation date %s done' % date.strftime('%Y-%m-%d'))

	session.commit()
	session.close()


def update_stock_factor_return(df_ret, last_date = None):

    if last_date is None:

        db = database.connection('asset')
        Session = sessionmaker(bind = db)
        session = Session()
        record = session.query(func.max(stock_factor_return.trade_date)).first()
        last_date = record[0].strftime('%Y-%m-%d') if record[0] is not None else '1900-01-01'
        session.commit()
        session.close()

    df_ret = df_ret[df_ret.index >= last_date]

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    session.query(stock_factor_return).filter(stock_factor_return.trade_date >= last_date).delete()
    session.commit()
    session.close()

    df_ret = df_ret.stack()
    df_ret = df_ret.reset_index()
    df_ret.columns = ['trade_date', 'sf_id', 'ret']
    df_ret = df_ret.set_index(['sf_id', 'trade_date'])

    db = database.connection('asset')
    t = Table('stock_factor_return', MetaData(bind=db), autoload = True)
    database.batch(db, t, df_ret, pd.DataFrame())


def update_stock_factor_specific_return(df_sret, last_date = None):

    if last_date is None:

        db = database.connection('asset')
        Session = sessionmaker(bind = db)
        session = Session()
        record = session.query(func.max(stock_factor_specific_return.trade_date)).first()
        last_date = record[0].strftime('%Y-%m-%d') if record[0] is not None else '1900-01-01'
        session.commit()
        session.close()

    df_sret = df_sret[df_sret.index >= last_date]

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    session.query(stock_factor_specific_return).filter(stock_factor_specific_return.trade_date >= last_date).delete()
    session.commit()
    session.close()

    df_sret = df_sret.stack()
    df_sret = df_sret.reset_index()
    df_sret.columns = ['trade_date', 'stock_id', 'sret']
    df_sret = df_sret.set_index(['stock_id', 'trade_date'])

    db = database.connection('asset')
    t = Table('stock_factor_specific_return', MetaData(bind=db), autoload = True)
    database.batch(db, t, df_sret, pd.DataFrame())


def update_exposure_mongo(sf):

    exposure = sf.exposure
    df_new = exposure.stack()
    df_new = df_new.reset_index()
    df_new['sf_id'] = sf.factor_id
    df_new.columns = ['trade_date', 'stock_id', 'exposure', 'sf_id']
    dic = df_new.to_dict(orient = 'index')
    dicv = dic.values()

    mg = MyMongoDB(config_mongo.exposure)
    mg.insert_many(dicv)


def load_exposure_mongo(sf_id = None, stock_id = None, begin_date = None, end_date = None):

    mg = MyMongoDB(config_mongo.exposure)

    dic = {}
    if sf_id:
        dic["sf_id"] = sf_id
    if stock_id:
        dic["stock_id"] = stock_id
    if begin_date or end_date:
        dic["trade_date"] = {}
    if begin_date:
        dic["trade_date"]["$gte"] = begin_date
    if end_date:
        dic["trade_date"]["$lte"] = end_date

    data = mg.find(dic)
    df = pd.DataFrame(list(data))

    return df


if __name__ == '__main__':

    #df1 = load_stock_factor_return()
    #df2 = load_stock_factor_specific_return()
    #set_trace()
    #update_exposure(StockFactor.SizeStockFactor(factor_id = 'SF.000001'))
    t = datetime.now()
    # df = load_exposure_mongo(begin_date = "2012-01-01")
    df = load_exposure_mongo(begin_date = datetime(2012, 1, 1))
    print datetime.now() - t
    set_trace()









