#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
from . import database
import MySQLdb
import config

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# base.ra_index_nav
#
def load_series(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_index_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date.label('date'),
        t1.c.ra_nav.label('nav'),
    ]

    s = select(columns).where(t1.c.ra_index_id == id_)

    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
    if mask is not None:
        if hasattr(mask, "__iter__") and not isinstance(mask, str):
            s = s.where(t1.c.ra_mask.in_(mask))
        else:
            s = s.where(t1.c.ra_mask == mask)

    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df['nav']

def load_ohlc(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_index_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date,
        t1.c.ra_open,
        t1.c.ra_high,
        t1.c.ra_low,
        t1.c.ra_nav.label('ra_close'),
    ]

    s = select(columns).where(t1.c.ra_index_id == id_)

    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
    if mask is not None:
        if hasattr(mask, "__iter__") and not isinstance(mask, str):
            s = s.where(t1.c.ra_mask.in_(mask))
        else:
            s = s.where(t1.c.ra_mask == mask)

    df = pd.read_sql(s, db, index_col = ['ra_date'], parse_dates=['ra_date'])

    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df

def load_ohlcav(id_, reindex=None, begin_date=None, end_date=None, mask=None):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('ra_index_nav', metadata, autoload=True)

    columns = [
        t1.c.ra_date,
        t1.c.ra_open,
        t1.c.ra_high,
        t1.c.ra_low,
        t1.c.ra_nav.label('ra_close'),
        t1.c.ra_amount,
        t1.c.ra_volume,
    ]

    s = select(columns).where(t1.c.ra_index_id == id_)

    if begin_date is not None:
        s = s.where(t1.c.ra_date >= begin_date)
    if end_date is not None:
        s = s.where(t1.c.ra_date <= end_date)
    if mask is not None:
        if hasattr(mask, "__iter__") and not isinstance(mask, str):
            s = s.where(t1.c.ra_mask.in_(mask))
        else:
            s = s.where(t1.c.ra_mask == mask)

    df = pd.read_sql(s, db, index_col = ['ra_date'], parse_dates=['ra_date'])
    if reindex is not None:
        df = df.reindex(reindex, method='pad')

    return df



def index_value(start_date, end_date, ra_index_id):
    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    date_sql = build_sql_trade_date_weekly(start_date, end_date)

    sql = "SELECT ra_date as date, ra_index_id, ra_nav FROM ra_index_nav, (%s) E WHERE ra_date = E.td_date and ra_index_id = %s ORDER BY ra_date" % (date_sql, ra_index_id)

    # sql = "select iv_index_id,iv_index_code,iv_time,iv_value,DATE_FORMAT(`iv_time`,'%%Y%%u') week from ( select * from index_value where iv_time>='%s' and iv_time<='%s' order by iv_time desc) as k group by iv_index_id,week order by week desc" % (start_date, end_date)


    logger.debug("index_value: " + sql)

    conn  = MySQLdb.connect(**config.db_base)
    df = pd.read_sql(sql, conn, index_col = ['date', 'ra_index_id'], parse_dates=['date'])
    conn.close()

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df


def build_sql_trade_date_weekly(start_date, end_date, include_end_date=True):
    if type(start_date) != str:
        start_date = start_date.strftime("%Y-%m-%d")

    if type(end_date) != str:
        end_date = end_date.strftime("%Y-%m-%d")

    if include_end_date:
        condition = "(td_type & 0x02 OR td_date = '%s')" % (end_date)
    else:
        condition = "(td_type & 0x02)"

    return "SELECT td_date FROM trade_dates WHERE td_date BETWEEN '%s' AND '%s' AND %s" % (start_date, end_date, condition)
