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


logger = logging.getLogger(__name__)

Base = declarative_base()

class ra_composite_asset(Base):

    __tablename__ = 'ra_composite_asset'

    globalid = Column(String, primary_key = True)
    ra_name = Column(String)
    ra_calc_type = Column(Integer)
    ra_begin_date = Column(Date)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class ra_composite_asset_nav(Base):

    __tablename__ = 'ra_composite_asset_nav'

    ra_asset_id = Column(String, primary_key = True)
    ra_date = Column(Date, primary_key = True)
    ra_nav = Column(Float)
    ra_inc = Column(Float)


def load_composite_asset_nav_series(asset_id, reindex=None, begin_date=None, end_date=None):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(ra_composite_asset_nav.ra_date , ra_composite_asset_nav.ra_nav).filter(ra_composite_asset_nav.ra_asset_id == asset_id)
    if begin_date is not None:
        sql = sql.filter(ra_composite_asset_nav.ra_date >= begin_date)
    if end_date is not None:
        sql = sql.filter(ra_composite_asset_nav.ra_date <= end_date)
    df = pd.read_sql(sql.statement, session.bind, index_col=['ra_date'], parse_dates=['ra_date'])
    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    ser = df.ra_nav
    ser.index.name = 'date'

    session.commit()
    session.close()

    return ser
