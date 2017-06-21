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

def get_monthly_data(m_start, m_end):
    db = database.connection('portfolio_sta')
    metadata = MetaData(bind=db)
    t = Table('ds_order', metadata, autoload=True)
    columns = [
        #func.max(t.c.ds_trade_date).label('newest_date')
        t.c.ds_uid,
        t.c.ds_portfolio_id,
        t.c.ds_trade_date,
        t.c.ds_trade_type,
        t.c.ds_amount,
    ]
    s = select(columns).where(t.c.ds_trade_date >= m_start) \
                        .where(t.c.ds_trade_date <= m_end)
    df = pd.read_sql(s, db)
    return df

def get_min_date():
    db = database.connection('portfolio_sta')
    metadata = MetaData(bind=db)
    t = Table('ds_order', metadata, autoload=True)
    Session = sessionmaker(bind=db)
    session = Session()
    rst = session.query(t).order_by(asc(t.c.ds_trade_date)).first()
    return rst.ds_trade_date

def get_max_date():
    db = database.connection('portfolio_sta')
    metadata = MetaData(bind=db)
    t = Table('ds_order', metadata, autoload=True)
    Session = sessionmaker(bind=db)
    session = Session()
    rst = session.query(t).order_by(desc(t.c.ds_trade_date)).first()
    return rst.ds_trade_date
def get_specific_month_data(s_date, e_date, t_type):
    """
    获取某个时间段内ds_trade_type=t_type的用户uid
    :param s_date: 开始日期
    :param e_date: 结束日期
    :param t_type: 交易类型
    :return: array(uid1, uid2)
    """
    db = database.connection('portfolio_sta')
    metadata = MetaData(bind=db)
    t = Table('ds_order', metadata, autoload=True)
    Session = sessionmaker(bind=db)
    session = Session()
    rst = session.query(t.c.ds_uid).filter(t.c.ds_trade_date >= s_date, \
                                            t.c.ds_trade_date <= e_date, \
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
    db = database.connection('portfolio_sta')
    metadata = MetaData(bind=db)
    t = Table('ds_order', metadata, autoload=True)
    Session = sessionmaker(bind=db)
    session = Session()
    rst = session.query(func.COUNT(func.DISTINCT(t.c.ds_uid))).filter( \
                                        t.c.ds_trade_date >= s_date, \
                                        t.c.ds_trade_date <= e_date, \
                                        t.c.ds_trade_type == t_type, \
                                        t.c.ds_uid.in_(uids))
    return rst.all()

if __name__ == "__main__":
    # get_mothly_data('2017-01-01', '2017-01-31')
    max_date = get_min_date()
    rst = get_specific_month_first_buy_num('2017-01-01', '2017-01-31', 10, [1000000006, 1000000001])
    print rst[0][0]
