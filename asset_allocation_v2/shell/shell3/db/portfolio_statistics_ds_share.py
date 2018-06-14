#coding=utf8

from sqlalchemy import MetaData, Table, select, func, desc, asc
from sqlalchemy.orm import sessionmaker
import pandas as pd
import datetime
import logging
from . import database
import os
import numpy as np
from dateutil.parser import parse
logger = logging.getLogger(__name__)
db = database.connection('portfolio_sta')
metadata = MetaData(bind=db)
Session = sessionmaker(bind=db)
session = Session()
def get_specific_date_amount(hold_date, uids):
    """
    获取某一天ds_uid in uids的用户持仓金额
    :param hold_date:持仓日期
    :param uids:用户id 列表
    :return: list
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_share', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(func.SUM(t.c.ds_amount)).filter(t.c.ds_date == hold_date, \
                                            t.c.ds_uid.in_(uids))

    return rst.all()
def get_specific_month_amount(hold_date):
    """
    获取某天用户总持仓金额
    :return: list
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_share', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(func.SUM(t.c.ds_amount)).filter( \
                                    t.c.ds_date == hold_date)
    return rst.all()

def get_aum_from_user_sta_table(hold_date):
    """
    从用户统计表里获取某天总的在管资产
    :return: list
    """
    t = Table('user_statistics', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(func.SUM(t.c.us_holding_amount)).filter( \
                                    t.c.us_date == hold_date)
    return rst.all()

def get_specific_month_hold_users(hold_date):
    """
    获取某天有持仓用户
    :return: list
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_share', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(t.c.ds_uid, func.sum(t.c.ds_amount)).filter( \
                                    t.c.ds_date == hold_date) \
                                    .group_by(t.c.ds_uid)
    return rst.all()

def get_specific_month_hold_count(hold_date):
    """
    获取某天有持仓用户
    :return: list
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_share', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(func.COUNT(func.DISTINCT(t.c.ds_uid))).filter( \
                                    t.c.ds_date == hold_date,
                                    t.c.ds_amount > 0)
    return rst.all()

def get_hold_users_date_uids(hold_date, uids):
    """
    获取某天有持仓用户且用户id在uids里
    :return: list
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_share', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(t.c.ds_uid, func.sum(t.c.ds_amount)).filter( \
                                    t.c.ds_date == hold_date, \
                                    t.c.ds_uid.in_(uids)) \
                                    .group_by(t.c.ds_uid)
    # rst = session.query(func.DISTINCT(t.c.ds_uid)).filter( \
    #                                 t.c.ds_date == hold_date,
    #                                 t.c.ds_amount > 0,
    #                                 t.c.ds_uid.in_(uids))
    return rst.all()
session.close()

if __name__ == "__main__":
    # get_mothly_data('2017-01-01', '2017-01-31')
    max_date = get_min_date()
    rst = get_specific_month_first_buy_num('2017-01-01', '2017-01-31', 10, [1000000006, 1000000001])
    print((rst[0][0]))
