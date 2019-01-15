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

db = database.connection('portfolio_sta')
metadata = MetaData(bind=db)
Session = sessionmaker(bind=db)
session = Session()
def get_monthly_holding_num(m_start, m_end):
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('user_statistics', metadata, autoload=True)
    #Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(t).filter( \
        t.c.us_date >= m_start, \
        t.c.us_date <= m_end) \
        .first()
    if rst is None:
        return 0
    num = rst.us_holding_users
    if num is None:
        min_date = 0
    return num
session.close()

if __name__ == "__main__":
    # get_mothly_data('2017-01-01', '2017-01-31')
    rst = get_monthly_holding_num('2018-01-31', '2018-01-31')
    print rst
