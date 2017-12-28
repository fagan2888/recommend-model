#coding=utf8


from sqlalchemy import Column, String, Integer, Text, Date, DateTime, Float
import logging
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()

class stock_factor(Base):

    __tablename__ = 'stock_factor'

    sf_id = Column(String, primary_key = True)
    sf_name = Column(String)
    sf_explain = Column(Text)
    sf_source = Column(Integer)
    sf_kind = Column(String)
    sf_formula = Column(String)
    sf_start_date = Column(Date)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)



class stock_factor_value(Base):

    __tablename__ = 'stock_factor_value'

    stock_id = Column(String, primary_key = True)
    sf_id = Column(String, primary_key = True)
    trade_date = Column(String, primary_key = True)
    factor_value = Column(Float)

    updated_at = Column(DateTime)
    created_at = Column(DateTime)
