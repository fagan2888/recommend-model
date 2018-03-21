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


#
# tc_timing
#
def load(id_, xtypes=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_bl', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.bl_name,
        t1.c.bl_type,
        t1.c.bl_method,
    ]

    s = select(columns)
    if id_ is not None:
        s = s.where(t1.c.globalid.in_(id_))
    if xtypes is not None:
        s = s.where(t1.c.tc_type.in_(xtypes))

    s = s.where(t1.c.bl_method != 0)
    df = pd.read_sql(s, db)

    return df



class ra_bl(Base):

    __tablename__ = 'ra_bl'

    globalid = Column(String, primary_key = True)
    bl_name = Column(String)
    bl_type = Column(Integer)
    bl_method = Column(Integer)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)


class ra_bl_view(Base):

    __tablename__ = 'ra_bl_view'

    globalid = Column(String, primary_key = True)
    bl_date = Column(String, primary_key = True)
    bl_index_id = Column(String, primary_key = True)
    bl_view = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)
