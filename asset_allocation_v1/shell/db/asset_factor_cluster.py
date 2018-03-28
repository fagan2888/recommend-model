#coding=utf8


from sqlalchemy import MetaData, Table, select, func
from sqlalchemy import Column, String, Integer, ForeignKey, Text, Date, DateTime, Float
import pandas as pd
import logging
import database
from sqlalchemy.ext.declarative import declarative_base
from dateutil.parser import parse


from dateutil.parser import parse


logger = logging.getLogger(__name__)


Base = declarative_base()


class factor_cluster(Base):

    __tablename__ = 'factor_cluster'

    globalid = Column(String, primary_key = True)
    fc_name = Column(String)
    fc_method = Column(String)
    fc_json_struct = Column(String)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class factor_cluster_argv(Base):

    __tablename__ = 'factor_cluster_argv'

    globalid = Column(String, primary_key = True)
    fc_key = Column(String)
    fc_value = Column(String)

    fc_value = Column(String)


class factor_cluster_asset(Base):

    __tablename__ = 'factor_cluster_asset'

    globalid = Column(String, primary_key = True)
    fc_asset_id = Column(String)


class factor_cluster_struct(Base):

    __tablename__ = 'factor_cluster_struct'

    globalid = Column(String, primary_key = True)
    fc_parent_cluster_id = Column(String, primary_key = True)
    fc_subject_asset_id = Column(String, primary_key = True)
    depth = Column(Integer)


class factor_cluster_nav(Base):

    __tablename__ = 'factor_cluster_nav'

    globalid = Column(String, primary_key = True)
    factor_cluster_id = Column(String, primary_key = True)
    date = Column(Date, primary_key = True)
    nav = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class barra_stock_factor_valid_factor(Base):

    __tablename__ = 'barra_stock_factor_valid_factor'

    bf_layer_id = Column(String, primary_key = True)
    trade_date = Column(Date, primary_key = True)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


def load_series(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('factor_cluster_nav', metadata, autoload=True)

    columns = [
        t1.c.date.label('date'),
        t1.c.nav.label('nav'),
    ]

    s = select(columns).where(t1.c.fc_cluster_id == id_)

    if begin_date is not None:
        s = s.where(t1.c.date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.date <= end_date)

    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['nav']
