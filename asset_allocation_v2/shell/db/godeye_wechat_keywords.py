#coding=utf8

from sqlalchemy import MetaData, Table, select, func, desc, asc
from sqlalchemy.orm import sessionmaker, Query
import pandas as pd
import datetime
import logging
from . import database
import os
import numpy as np
from dateutil.parser import parse
logger = logging.getLogger(__name__)
db = database.connection('godeye')
metadata = MetaData(bind=db)
Session = sessionmaker(bind=db)
session = Session()
def get_min_date():
    t = Table('wechat_keywords', metadata, autoload=True)
    rst = session.query(t).order_by(asc(t.c.wk_date)).first()
    if rst is None:
        return rst
    return rst.wk_date

def get_max_date():
    """
    为了增量更新获取当前所属月份（最新）
    """
    t = Table('wechat_keywords', metadata, autoload=True)
    rst = session.query(t).order_by(desc(t.c.wk_date)).first()
    if rst is None:
        return rst
    return rst.wk_date


def get_old_data(cur_date):
    """
    得到所属月份数据
    :param dates: list, 所属月份列表
    :return: df
    """
    t = Table('wechat_keywords', metadata, autoload=True)
    columns = [
        #func.max(t.c.ds_trade_date).label('newest_date')
        t.c.wk_date,
        t.c.wk_keywords,
        t.c.wk_times,
        t.c.wk_type,
    ]
    rst = session.query(
        t.c.wk_date, \
        t.c.wk_keywords, \
        t.c.wk_times, \
        t.c.wk_type).filter( \
        t.c.wk_date == cur_date)
    # rst = session.query(t).filter( \
    #     t.c.wk_date == cur_date)
    return rst.all()

def batch(df_new, df_old):
    t = Table('wechat_keywords', metadata, autoload=True)
    fmt_columns = []
    fmt_precision = 4
    if not df_new.empty:
        df_new = database.number_format(df_new, fmt_columns, fmt_precision)
    if not df_old.empty:
        df_old = database.number_format(df_old, fmt_columns, fmt_precision)
    database.batch(db, t, df_new, df_old)

# def save(gid, xtype, df):
#     fmt_columns = ['rp_user_redeem_ratio', 'ra_inc']
#     fmt_precision = 6
#     if not df.empty:
#         df = database.number_format(df, fmt_columns, fmt_precision)
#     #
#     # 保存择时结果到数据库
#     #
#     db = database.connection('asset')
#     t2 = Table('ra_portfolio_nav', MetaData(bind=db), autoload=True)
#     columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
#     s = select(columns, (t2.c.ra_portfolio_id == gid)).where(t2.c.ra_type == xtype)
#     df_old = pd.read_sql(s, db, index_col=['ra_portfolio_id', 'ra_type', 'ra_date'], parse_dates=['ra_date'])
#     if not df_old.empty:
#         df_old = database.number_format(df_old, fmt_columns, fmt_precision)

#     # 更新数据库
#     # print df_new.head()
#     # print df_old.head()
#     database.batch(db, t2, df, df_old, timestamp=True)


session.close()

if __name__ == "__main__":
    # get_mothly_data('2017-01-01', '2017-01-31')
    print(get_max_date())
