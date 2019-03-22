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
def get_monthly_data(m_start, m_end):
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_order_pdate', metadata, autoload=True)
    columns = [
        #func.max(t.c.ds_trade_date).label('newest_date')
        t.c.ds_uid,
        t.c.ds_portfolio_id,
        t.c.ds_placed_date,
        t.c.ds_trade_type,
        t.c.ds_amount,
    ]
    s = select(columns).where(t.c.ds_placed_date >= m_start) \
                        .where(t.c.ds_placed_date <= m_end)
    df = pd.read_sql(s, db)
    return df
def get_date_range_head(s_date, e_date, uid = None, htype = 1):
    """
    获取时间区间内最新或者最老一条记录
    :param s_date:处理记录开始时间
    :param e_date:处理记录结束时间
    :param uid:用户id
    :param htype: 0:最老记录，1:最新记录
    :return: sqlalchemy result
    """
    t = Table('ds_order_pdate', metadata, autoload=True)
    if htype == 0:
        rst = session.query(t).filter(t.c.ds_uid == uid, \
            t.c.ds_placed_date >= s_date, \
            t.c.ds_placed_date <= e_date) \
            .order_by(asc(t.c.ds_placed_date)) \
            .order_by(asc(t.c.ds_placed_time)) \
            .first()
    elif htype == 1:
        rst = session.query(t).filter(t.c.ds_uid == uid, \
            t.c.ds_placed_date >= s_date, \
            t.c.ds_placed_date <= e_date) \
            .order_by(desc(t.c.ds_placed_date)) \
            .order_by(desc(t.c.ds_placed_time)) \
            .first()
    else:
        return None
    return rst
def get_min_date():
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_order_pdate', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(t).order_by(asc(t.c.ds_placed_date)).order_by(asc(t.c.ds_placed_time)).first()
    min_date = rst.ds_placed_date
    if min_date is None:
        min_date = datetime.date(2016, 8, 17)
    return min_date

def get_max_date():
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_order_pdate', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(t).order_by(desc(t.c.ds_placed_date)).first()
    max_date = rst.ds_placed_date
    if max_date is None:
        max_date = datetime.datetime.now().strftime('%Y-%m-%d')
    return rst.ds_placed_date
def get_specific_month_data(s_date, e_date, t_type):
    """
    获取某个时间段内ds_trade_type=t_type的用户uid
    :param s_date: 开始日期
    :param e_date: 结束日期
    :param t_type: 交易类型
    :return: array(uid1, uid2)
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_order_pdate', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(t.c.ds_uid).filter(t.c.ds_placed_date >= s_date, \
                                            t.c.ds_placed_date <= e_date, \
                                            t.c.ds_trade_type == t_type)
    return rst.all()
def get_specific_month_num(s_date, e_date, t_type, uids):
    """
    获取某个时间段内ds_trade_type=t_type的且uid在uids内的用户数
    :param s_date: string, 开始日期
    :param e_date: string, 结束日期
    :param t_type: int, 交易类型
    :param uids: array like,  用户uid
    :return: int
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_order_pdate', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(func.COUNT(func.DISTINCT(t.c.ds_uid))).filter( \
                                        t.c.ds_placed_date >= s_date, \
                                        t.c.ds_placed_date <= e_date, \
                                        t.c.ds_trade_type == t_type, \
                                        t.c.ds_uid.in_(uids))
    return rst.all()

def get_specific_month_num_naive(s_date, e_date, t_type):
    """
    获取某个时间段内ds_trade_type=t_type的且uid在uids内的用户数
    :param s_date: string, 开始日期
    :param e_date: string, 结束日期
    :param t_type: array, 交易类型
    :return: int
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_order_pdate', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(func.COUNT(func.DISTINCT(t.c.ds_uid))).filter( \
                                        t.c.ds_placed_date >= s_date, \
                                        t.c.ds_placed_date <= e_date, \
                                        t.c.ds_trade_type.in_(t_type))
    return rst.all()

def get_specific_month_in_uids(s_date, e_date, t_type, uids):
    """
    获取某个时间段内ds_trade_type=t_type的且uid在uids内的用户uid
    :param s_date: string, 开始日期
    :param e_date: string, 结束日期
    :param t_type: list, 交易类型
    :param uids: array like,  用户uid
    :return: int
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_order_pdate', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(func.DISTINCT(t.c.ds_uid)).filter( \
                                        t.c.ds_placed_date >= s_date, \
                                        t.c.ds_placed_date <= e_date, \
                                        t.c.ds_trade_type.in_(t_type), \
                                        t.c.ds_uid.in_(uids))
    return rst.all()

def get_specific_month_amount(s_date, e_date, t_type, uids):
    """
    获取某个时间段内ds_trade_type=t_type的复购总金额
    :param s_date: string, 开始日期
    :param e_date: string, 结束日期
    :param t_type: list, 交易类型
    :return: list
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_order_pdate', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(func.SUM(t.c.ds_amount)).filter( \
                                        t.c.ds_placed_date >= s_date, \
                                        t.c.ds_placed_date <= e_date, \
                                        t.c.ds_trade_type.in_(t_type), \
                                        t.c.ds_uid.in_(uids))
    # session.close()
    return rst.all()

def get_specific_day_amount(date, t_type):
    """
    获取某个时间段内ds_trade_type=t_type的总金额
    :param date: string, 日期
    :param t_type: list, 交易类型
    :return: list
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_order_pdate', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(func.SUM(t.c.ds_amount)).filter( \
                                        t.c.ds_placed_date == date, \
                                        t.c.ds_trade_type.in_(t_type))
    # session.close()
    return rst.all()
def get_specific_month_uids(s_date, e_date, t_type):
    """
    获取某个时间段内ds_trade_type=t_type的用户id
    :param s_date: string, 开始日期
    :param e_date: string, 结束日期
    :param t_type: int, 交易类型
    :return: list
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_order_pdate', metadata, autoload=True)
    #Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(func.DISTINCT(t.c.ds_uid)).filter( \
                                        t.c.ds_placed_date >= s_date, \
                                        t.c.ds_placed_date <= e_date, \
                                        t.c.ds_trade_type == t_type)
    #session.close()
    return rst.all()

def get_specific_month_uids_in(s_date, e_date, t_type):
    """
    获取某个时间段内ds_trade_type=t_type的用户id
    :param s_date: string, 开始日期
    :param e_date: string, 结束日期
    :param t_type: list, 交易类型
    :return: list
    """
    # db = database.connection('portfolio_sta')
    # metadata = MetaData(bind=db)
    t = Table('ds_order_pdate', metadata, autoload=True)
    #Session = sessionmaker(bind=db)
    # session = Session()
    rst = session.query(func.DISTINCT(t.c.ds_uid)).filter( \
                                        t.c.ds_placed_date >= s_date, \
                                        t.c.ds_placed_date <= e_date, \
                                        t.c.ds_trade_type.in_(t_type))
    #session.close()
    return rst.all()
session.close()

if __name__ == "__main__":
    # get_mothly_data('2017-01-01', '2017-01-31')
    max_date = get_min_date()
    rst = get_specific_month_first_buy_num('2017-01-01', '2017-01-31', 10, [1000000006, 1000000001])
    print rst[0][0]
