#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import sys
sys.path.append('shell')
from sqlalchemy import MetaData, Table, select, func, and_
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
import numpy as np
import pandas as pd
import logging
from . import database
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from ipdb import set_trace

logger = logging.getLogger(__name__)

Base = declarative_base()


class fund_pos(Base):

    __tablename__ = 'fund_pos'

    fund_id = Column(Integer, primary_key=True)
    index_id = Column(String, primary_key=True)
    trade_date = Column(Date, primary_key=True)
    position = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


def load_fund_pos(fund_id=None, fund_ids=None, index_id=None, index_ids=None, begin_date=None, end_date=None):

    db = database.connection('asset')
    Session = sessionmaker(bind=db)
    session = Session()
    record = session.query(
        fund_pos.fund_id,
        fund_pos.index_id,
        fund_pos.trade_date,
        fund_pos.position,
        )

    if fund_id:
        record = record.filter(fund_pos.fund_id == fund_id)
    if fund_ids is not None:
        record = record.filter(fund_pos.fund_id.in_(fund_ids))
    if index_id:
        record = record.filter(fund_pos.index_id == index_id)
    if index_id is not None:
        record = record.filter(fund_pos.index_id.in_(index_ids))
    if begin_date:
        record = record.filter(fund_pos.trade_date >= begin_date)
    if end_date:
        record = record.filter(fund_pos.trade_date <= end_date)

    df = pd.read_sql(record.statement, session.bind, index_col=['fund_id', 'index_id', 'trade_date'], parse_dates=['trade_date'])
    session.commit()
    session.close()

    return df


def update_fund_pos(df_new, last_date=None):

    df_old = load_fund_pos()
    db = database.connection('asset')
    t = Table('fund_pos', MetaData(bind=db), autoload=True)
    database.batch(db, t, df_new, df_old)
