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
def get_min_date():
    t = Table('rpt_srrc_rolling', metadata, autoload=True)
    #rst = session.query(t).order_by(asc(t.c.ds_trade_date)).first()
    #return rst.ds_trade_date

def get_max_date():
    """
    为了增量更新获取当前所属月份（最新）
    """
    t = Table('rpt_srrc_rolling', metadata, autoload=True)
    rst = session.query(t).order_by(desc(t.c.rp_date)).first()
    return rst
def get_old_data(dates):
    """
    得到所属月份数据
    :param dates: list, 所属月份列表
    :return: df
    """
    t = Table('rpt_srrc_rolling', metadata, autoload=True)
    columns = [
        #func.max(t.c.ds_trade_date).label('newest_date')
        t.c.rp_tag_id,
        t.c.rp_date,
        t.c.rp_rolling_window,
        t.c.rp_user_redeem_ratio,
        t.c.rp_user_resub_ratio,
        t.c.rp_amount_redeem_ratio,
        t.c.rp_amount_resub_ratio,
    ]
    rst = session.query(t).filter(t.c.rp_date.in_(dates))
    return rst.all()

def batch(df_new, df_old):
    t = Table('rpt_srrc_rolling', metadata, autoload=True)
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
    print get_max_date()
