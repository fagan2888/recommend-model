#coding=utf8

from sqlalchemy import MetaData, Table, select, func, desc, asc
from sqlalchemy.orm import sessionmaker
import pandas as pd
import datetime
import logging
import database
import os
import numpy as np
from dateutil.parser import parse
logger = logging.getLogger(__name__)

def get_monthly_data(m_start, m_end):
    db = database.connection('portfolio_sta')
    metadata = MetaData(bind=db)
    t = Table('ds_order', metadata, autoload=True)
    columns = [
        #func.max(t.c.ds_trade_date).label('newest_date')
        t.c.ds_uid,
        t.c.ds_portfolio_id,
        t.c.ds_trade_date,
        t.c.ds_trade_type,
        t.c.ds_amount,
    ]
    s = select(columns).where(t.c.ds_trade_date >= m_start) \
                        .where(t.c.ds_trade_date <= m_end)
    df = pd.read_sql(s, db)
    return df

def get_min_date():
    db = database.connection('portfolio_sta')
    metadata = MetaData(bind=db)
    t = Table('ds_order', metadata, autoload=True)
    Session = sessionmaker(bind=db)
    session = Session()
    rst = session.query(t).order_by(asc(t.c.ds_trade_date)).first()
    return rst.ds_trade_date

def get_max_date():
    db = database.connection('portfolio_sta')
    metadata = MetaData(bind=db)
    t = Table('ds_order', metadata, autoload=True)
    columns = [
        func.max(t.c.ds_trade_date).label('newest_date'),
    ]
    s = select(columns)
    Session = sessionmaker(bind=db)
    session = Session()
    rst = session.query(t).order_by(desc(t.c.ds_trade_date)).first()
    return rst.ds_trade_date

if __name__ == "__main__":
    # get_mothly_data('2017-01-01', '2017-01-31')
    max_date = get_min_date()
    print max_date
