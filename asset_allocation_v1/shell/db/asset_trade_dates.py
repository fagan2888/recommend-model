#coding=utf8

from sqlalchemy import MetaData, Table, select, func
import pandas as pd
import logging
import database
from dateutil.parser import parse
import sys
logger = logging.getLogger(__name__)
def load_trade_dates():
    db = database.connection('asset')
    print db
    metadata = MetaData(bind=db)
    t = Table('trade_dates', metadata, autoload=True)

    columns = [
        t.c.td_date.label('date'),
        t.c.td_type.label('trade_type'),
    ]
    s = select(columns)
    s = s.order_by(t.c.td_date.asc())
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])
    return df
