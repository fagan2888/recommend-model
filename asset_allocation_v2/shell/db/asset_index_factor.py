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


class index_factor_exposure(Base):

    __tablename__ = 'index_factor_exposure'

    index_id = Column(String, primary_key = True)
    if_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)
    exposure = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


def load_index_factor_exposure(index_id = None, index_ids = None, if_id = None, if_ids = None, begin_date = None, end_date = None):

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    record = session.query(
        index_factor_exposure.index_id,
        index_factor_exposure.if_id,
        index_factor_exposure.trade_date,
        index_factor_exposure.exposure,
        )

    if index_id:
        record = record.filter(index_factor_exposure.index_id == index_id)
    if index_ids is not None:
        record = record.filter(index_factor_exposure.index_id.in_(index_ids))
    if if_id:
        record = record.filter(index_factor_exposure.if_id == if_id)
    if if_ids is not None:
        record = record.filter(index_factor_exposure.if_id.in_(if_ids))
    if begin_date:
        record = record.filter(index_factor_exposure.trade_date >= begin_date)
    if end_date:
        record = record.filter(index_factor_exposure.trade_date <= end_date)

    df = pd.read_sql(record.statement, session.bind, index_col = ['index_id', 'if_id', 'trade_date'], parse_dates = ['trade_date'])
    session.commit()
    session.close()

    return df



def update_exposure(indexfactor, last_date = None):

    if_id = indexfactor.factor_id
    exposure = indexfactor.exposure

    if last_date is None:

        db = database.connection('asset')
        Session = sessionmaker(bind = db)
        session = Session()
        record = session.query(func.max(index_factor_exposure.trade_date)).filter(index_factor_exposure.if_id == if_id).first()
        last_date = record[0].strftime('%Y-%m-%d') if record[0] is not None else '1900-01-01'
        session.commit()
        session.close()

    exposure = exposure[exposure.index >= last_date]

    db = database.connection('asset')
    Session = sessionmaker(bind = db)
    session = Session()
    session.query(index_factor_exposure).filter(index_factor_exposure.trade_date >= last_date).filter(index_factor_exposure.if_id == if_id).delete()
    session.commit()
    session.close()

    df_new = exposure.stack()
    df_new = df_new.reset_index()
    df_new['if_id'] = indexfactor.factor_id
    df_new.columns = ['trade_date', 'index_id', 'exposure', 'if_id']
    df_new = df_new.set_index(['index_id', 'if_id', 'trade_date'])

    db = database.connection('asset')
    t = Table('index_factor_exposure', MetaData(bind=db), autoload = True)
    database.batch(db, t, df_new, pd.DataFrame())



if __name__ == '__main__':

    #df1 = load_fund_factor_return()
    #df2 = load_fund_factor_specific_return()
    #set_trace()
    #update_exposure(fundFactor.SizefundFactor(factor_id = 'ff.000001'))
    set_trace()









