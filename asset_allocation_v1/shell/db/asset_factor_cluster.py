#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

from sqlalchemy.orm import sessionmaker,relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy import *

Base = declarative_base()

class factor_cluster(Base):

    __tablename__ = 'factor_cluster'

    globalid = Column(String, primary_key=True)
    fc_name = Column(String)
    fc_method = Column(Integer)
    fc_json_struct = Column(Text)


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
    fc_cluster_id = Column(String, primary_key = True)
    date = Column(Date)
    nav = Column(Float)
