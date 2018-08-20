#coding=utf8

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
import asset

logger = logging.getLogger(__name__)

Base = declarative_base()


class index_factor(Base):

    __tablename__ = 'index_factor'

    index_id = Column(String, primary_key = True)
    if_type = Column(Integer)
    secode = Column(String)
    index_name = Column(String)


def load_all_index_factor(if_type = None):

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(index_factor.index_id, index_factor.secode, index_factor.index_name)
    if if_type is not None:
        sql = sql.filter(index_factor.if_type == if_type)
    else:
        sql = sql.filter(index_factor.if_type > 0.0)
    all_indexes = pd.read_sql(sql.statement, session.bind, index_col = ['index_id'])
    session.commit()
    session.close()

    return all_indexes




if __name__ == '__main__':

    pass
