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

def get_min_date():
    db = database.connection('portfolio_sta')
    metadata = MetaData(bind=db)
    t = Table('rpt_srrc_apportion', metadata, autoload=True)
    Session = sessionmaker(bind=db)
    session = Session()
    #rst = session.query(t).order_by(asc(t.c.ds_trade_date)).first()
    #return rst.ds_trade_date

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

if __name__ == "__main__":
    # get_mothly_data('2017-01-01', '2017-01-31')
    print 'main'
