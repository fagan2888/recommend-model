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
def get_min_date():
    t = Table('rpt_retention_data', metadata, autoload=True)
    #rst = session.query(t).order_by(asc(t.c.ds_trade_date)).first()
    #return rst.ds_trade_date

def get_max_date():
    """
    为了增量更新获取当前所属月份（最新）
    """
    t = Table('rpt_retention_data', metadata, autoload=True)
    rst = session.query(t).order_by(desc(t.c.rp_date)).first()
    return rst
def get_old_data(dates):
    """
    得到所属月份数据
    :param dates: list, 所属月份列表
    :return: df
    """
    t = Table('rpt_retention_data', metadata, autoload=True)
    columns = [
        #func.max(t.c.ds_trade_date).label('newest_date')
        t.c.rp_tag_id,
        t.c.rp_date,
        t.c.rp_retention_type,
        t.c.rp_user_resub,
        t.c.rp_user_hold,
        t.c.rp_amount_resub,
        t.c.rp_amount_redeem,
        t.c.rp_amount_aum,
    ]
    rst = session.query(t).filter(t.c.rp_date.in_(dates))
    return rst.all()
def batch(df_new, df_old):
    t = Table('rpt_retention_data', metadata, autoload=True)
    database.batch(db, t, df_new, df_old)
session.close()

if __name__ == "__main__":
    # get_mothly_data('2017-01-01', '2017-01-31')
    print(get_max_date())
