#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

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
import numpy as np
import time


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

