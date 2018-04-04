#coding=utf8


from sqlalchemy import Column, String, Integer, Text, Date, DateTime, Float
import logging
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from db.asset_stock_factor import *
from db.asset_stock import *
from db import asset_trade_dates


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



class stock_factor_value(Base):

    __tablename__ = 'stock_factor_value'

    stock_id = Column(String, primary_key = True)
    sf_id = Column(String, primary_key = True)
    trade_date = Column(String, primary_key = True)
    factor_value = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class stock_factor_stock_valid(Base):

    __tablename__ = 'stock_factor_stock_valid'

    stock_id = Column(String, primary_key = True)
    secode   = Column(String)
    trade_date = Column(String, primary_key = True)
    valid = Column(Integer)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class stock_factor_rankcorr(Base):

    __tablename__ = 'stock_factor_rankcorr'

    sf_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    rankcorr = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class stock_factor_layer(Base):

    __tablename__ = 'stock_factor_layer'

    sf_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    layer = Column(Integer, primary_key = True)
    stock_ids = Column(Text)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class stock_factor_nav(Base):

    __tablename__ = 'stock_factor_nav'

    sf_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    nav = Column(Integer)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class barra_stock_factor_exposure(Base):

    __tablename__ = 'barra_stock_factor_exposure'

    stock_id = Column(String, primary_key = True)
    bf_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    factor_exposure = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class barra_stock_factor_layer_stocks(Base):

    __tablename__ = 'barra_stock_factor_layer_stocks'

    bf_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    layer = Column(Integer, primary_key = True)
    stock_ids = Column(Text)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class barra_stock_factor_layer_nav(Base):

    __tablename__ = 'barra_stock_factor_layer_nav'

    bf_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    layer = Column(Integer, primary_key = True)
    nav = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class barra_stock_factor_selected_factor_nav(Base):

    __tablename__ = 'barra_stock_factor_selected_factor_nav'

    bf_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    selected_date = Column(Date, primary_key = True)
    nav = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class barra_stock_factor_layer_ic(Base):

    __tablename__ = 'barra_stock_factor_layer_ic'

    bf_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    ic = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class barra_stock_factor_layer_weight_nav(Base):

    __tablename__ = 'barra_stock_factor_layer_weight_nav'

    bf_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    layer = Column(Integer, primary_key = True)
    nav = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)



class barra_stock_factor_stock_allocate(Base):

    __tablename__ = 'barra_stock_factor_stock_allocate'

    trade_date = Column(Date, primary_key = True)
    stock_id = Column(String, primary_key = True)
    weight = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)



class barra_stock_factor_allocate_nav(Base):

    __tablename__ = 'barra_stock_factor_allocate_nav'

    allocate_date = Column(Date, primary_key = True)
    nav_date = Column(Date, primary_key = True)
    nav = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class barra_stock_factor_regression_tree(Base):

    __tablename__ = 'barra_stock_factor_regression_tree'

    trade_date = Column(Date, primary_key = True)
    tree = Column(Text)
    clusters = Column(Text)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class barra_stock_factor_valid_factor(Base):

    __tablename__ = 'barra_stock_factor_valid_factor'

    trade_date = Column(Date, primary_key = True)
    bf_layer_id = Column(String, primary_key = True)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


def load_factor_nav_series(asset_id, reindex=None, begin_date=None, end_date=None):

    bf_ids = asset_id.strip().split('.')
    layer = int(bf_ids[-1])
    bf_id = '.'.join(bf_ids[0:2])

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(barra_stock_factor_layer_nav.trade_date ,barra_stock_factor_layer_nav.nav).filter(and_(barra_stock_factor_layer_nav.bf_id == bf_id, barra_stock_factor_layer_nav.layer == layer))
    if begin_date is not None:
        sql = sql.filter(barra_stock_factor_layer_nav.trade_date >= begin_date)
    if end_date is not None:
        sql = sql.filter(barra_stock_factor_layer_nav.trade_date <= end_date)
    df = pd.read_sql(sql.statement, session.bind, index_col=['trade_date'], parse_dates=['trade_date'])
    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    ser = df.nav
    ser.index.name = 'date'
    session.commit()
    session.close()

    return ser


def load_selected_factor_nav_series(asset_id, reindex=None, begin_date=None, end_date=None):

    #bf_ids = asset_id.strip().split('.')
    #layer = int(bf_ids[-1])
    #bf_id = '.'.join(bf_ids[0:2])

    bf_id = asset_id

    month_last_trade_date_df = asset_trade_dates.load_month_last_trade_date()
    month_last_trade_date_df = month_last_trade_date_df[month_last_trade_date_df.index <= end_date]
    month_last_trade_date_df = month_last_trade_date_df.sort_index(ascending = True)
    factor_selected_date = month_last_trade_date_df.index[-1]


    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = session.query(barra_stock_factor_selected_factor_nav.trade_date ,barra_stock_factor_selected_factor_nav.nav).filter(and_(barra_stock_factor_selected_factor_nav.bf_id == bf_id, barra_stock_factor_selected_factor_nav.selected_date == factor_selected_date))

    if begin_date is not None:
        sql = sql.filter(barra_stock_factor_selected_factor_nav.trade_date >= begin_date)

    if end_date is not None:
        sql = sql.filter(barra_stock_factor_selected_factor_nav.trade_date <= end_date)

    df = pd.read_sql(sql.statement, session.bind, index_col=['trade_date'], parse_dates=['trade_date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    ser = df.nav
    ser.index.name = 'date'
    session.commit()
    session.close()

    return ser

