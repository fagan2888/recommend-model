#coding=utf8


import sys
sys.path.append('shell')
from sqlalchemy import MetaData, Table, select, func, and_
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
import pandas as pd
import MySQLdb
import logging
from . import database
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dateutil.parser import parse
import time
import asset
import numpy as np
from functools import reduce
from ipdb import set_trace

logger = logging.getLogger(__name__)

Base = declarative_base()

class mz_markowitz_bounds(Base):

    __tablename__ = 'mz_markowitz_bounds'

    globalid = Column(String, primary_key = True)
    mz_asset_id = Column(String)
    mz_asset_name = Column(String)
    mz_allocate_start_date = Column(Date)
    mz_allocate_end_date = Column(Date)
    mz_upper_limit = Column(Float)
    mz_lower_limit = Column(Float)
    mz_sum1_limit = Column(Float)
    mz_sum2_limit = Column(Float)
    mz_lower_sum1_limit = Column(Float)
    mz_lower_sum2_limit = Column(Float)
    mz_upper_sum1_limit = Column(Float)
    mz_upper_sum2_limit = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class rm_riskmgr_signal(Base):

    __tablename__ = 'rm_riskmgr_signal'

    rm_riskmgr_id = Column(String, primary_key = True)
    rm_date = Column(Date)
    rm_pos = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class rm_riskmgr_ecdf_vars(Base):

    __tablename__ = 'rm_riskmgr_ecdf_vars'

    index_id = Column(String, primary_key = True)
    ix_date = Column(Date, primary_key = True)
    ecdf_var = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class rm_riskmgr_index_best_start_end(Base):

    __tablename__ = 'rm_riskmgr_index_best_start_end'

    index_id = Column(String, primary_key = True)
    ix_date = Column(Date, primary_key = True)
    start = Column(Integer)
    end = Column(Integer)


    updated_at = Column(DateTime)
    created_at = Column(DateTime)



def load_mz_markowitz_bounds(globalid):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(mz_markowitz_bounds.mz_asset_id, mz_markowitz_bounds.mz_asset_name, mz_markowitz_bounds.mz_allocate_start_date,
            mz_markowitz_bounds.mz_allocate_end_date, mz_markowitz_bounds.mz_upper_limit, mz_markowitz_bounds.mz_lower_limit,
            mz_markowitz_bounds.mz_sum1_limit, mz_markowitz_bounds.mz_sum2_limit, mz_markowitz_bounds.mz_lower_sum1_limit, mz_markowitz_bounds.mz_lower_sum2_limit,
            mz_markowitz_bounds.mz_upper_sum1_limit, mz_markowitz_bounds.mz_upper_sum2_limit,
            ).filter(mz_markowitz_bounds.globalid == globalid)
    df = pd.read_sql(sql.statement, session.bind, index_col = ['mz_asset_id'], parse_dates = ['mz_allocate_start_date','mz_allocate_end_date'])
    session.commit()
    session.close()

    return df




if __name__ == '__main__':

    df = load_mz_markowitz_bounds('AB.000001')
    print(df.tail())
