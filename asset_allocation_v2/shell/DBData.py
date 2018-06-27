#coding=utf8


import string
# import MySQLdb
import config
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import sys
import logging
sys.path.append('shell')
import Const

from Const import datapath
from dateutil.parser import parse

from sqlalchemy import *
from db import database

logger = logging.getLogger(__name__)

#读取A股交易日
def all_trade_dates():

    db = database.connection('base')
    t = Table('trade_dates', MetaData(bind=db), autoload=True)
    s = select([t.c.td_date]).where(t.c.td_date >= '2002-01-04').where(t.c.td_type.op('&')(0x02)).order_by(t.c.td_date.asc())

    res = s.execute().fetchall()
    return [i[0].strftime('%Y-%m-%d') for i in res]

    # sql = "SELECT td_date FROM trade_dates WHERE td_date >= '2002-01-04' AND td_type & 0x02 ORDER by td_date ASC";


#交易日
def trade_dates(start_date, end_date):
    db = database.connection('base')
    t = Table('trade_dates', MetaData(bind=db), autoload=True)

    if not end_date:
        s = select([func.max(t.c.td_date).label('td_date')]).where(t.c.td_type.op('&')(0x02))
        end_date = db_pluck(s, 'td_date')

    s = select([t.c.td_date]).where(t.c.td_date.between(start_date, end_date)).where(t.c.td_type.op('&')(0x02) | (t.c.td_date == end_date)).order_by(t.c.td_date.asc())

    df = pd.read_sql(s, db, index_col='td_date')
    return df.index

    # if not end_date:
        # sql = "SELECT max(td_date) as td_date FROM trade_dates WHERE td_type & 0x02"
        # end_date = db_pluck(conn, 'td_date', sql)

    # sql = "SELECT td_date FROM trade_dates WHERE td_date BETWEEN '%s' AND '%s' AND (td_type & 0x02 OR td_date = '%s') ORDER By td_date ASC" % (start_date, end_date, end_date);


def db_pluck(s, col):
    return s.execute().fetchall()[0][col]

# def db_pluck(conn, col, sql):
    # cur = conn.cursor(MySQLdb.cursors.DictCursor)
    # cur.execute(sql)
    # records = cur.fetchall()

    # if records:
        # return records[0][col]
    # return None

def trade_date_index(start_date, end_date=None):
    db = database.connection('base')
    t = Table('trade_dates', MetaData(bind=db), autoload=True)

    if not end_date:
        s = select([func.max(t.c.td_date).label('td_date')]).where(t.c.td_date != func.curdate())
        end_date = db_pluck(s, 'td_date')

    s = select([t.c.td_date.label('date')]).where(t.c.td_date.between(start_date, end_date)).where((t.c.td_type.op('&')(0x02)) | (t.c.td_date == end_date)).order_by(t.c.td_date.asc())

    df = pd.read_sql(s, db, index_col='date', parse_dates=['date'])
    return df.index

    # if not end_date:
        # sql = "SELECT max(td_date) as td_date FROM trade_dates WHERE td_date != CURDATE()"
        # end_date = db_pluck(conn, 'td_date', sql)

    # sql = "SELECT td_date as date FROM trade_dates WHERE td_date BETWEEN '%s' AND '%s' AND (td_type & 0x02 OR td_date = '%s') ORDER By td_date ASC" % (start_date, end_date, end_date);


def trade_date_lookback_index(end_date=None, lookback=26, include_end_date=True):
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t1 = Table('trade_dates', metadata, autoload=True)
    columns = [
        t1.c.td_date.label('date'),
        t1.c.td_type,
    ]

    s = select(columns).where(t1.c.td_date <= end_date)
    if include_end_date:
        condition = s.where((t1.c.td_type.op('&')(0x02)) | (t1.c.td_date == end_date))
    else:
        condition = s.where((t1.c.td_type.op('&')(0x02)))

    s = condition.order_by(t1.c.td_date.desc()).limit(lookback)

    df = pd.read_sql(s, db, index_col = 'date', parse_dates=True)
    return df.index.sort_values()

    # if include_end_date:
        # condition = "(td_type & 0x02 OR td_date = '%s')" % (end_date)
    # else:
        # condition = "(td_type & 0x02)"

    # sql = "SELECT td_date as date, td_type FROM trade_dates WHERE td_date <= '%s' AND %s ORDER By td_date DESC LIMIT %d" % (end_date, condition, lookback)



def build_sql_trade_date_weekly(start_date, end_date, include_end_date=True):
    # if not(isinstance(start_date, str)):
        # start_date = start_date.strftime("%Y-%m-%d")
    # if not(isinstance(end_date, str)):
        # start_date = end_date.strftime("%Y-%m-%d")
    db = database.connection('base')
    t = Table('trade_dates', MetaData(bind=db), autoload=True)
    s = select([t.c.td_date]).where(t.c.td_date.between(start_date, end_date))
    if include_end_date:
        condition = s.where(t.c.td_type.op('&')(0x02) | (t.c.td_date == end_date))
    else:
        condition = s.where(t.c.td_type.op('&')(0x02))
    # return str(condition.compile(compile_kwargs={'literal_binds':True})).replace('\n', '')
    return condition

    # return "SELECT td_date FROM trade_dates WHERE td_date BETWEEN '%s' AND '%s' AND %s" % (start_date, end_date, condition)


def build_sql_trade_date_daily(start_date, end_date):
    # if not(isinstance(start_date, str)):
        # start_date = start_date.strftime("%Y-%m-%d")
    # if not(isinstance(end_date, str)):
        # start_date = end_date.strftime("%Y-%m-%d")
    db = database.connection('base')
    t = Table('trade_dates', MetaData(bind=db), autoload=True)
    s = select([t.c.td_date]) \
            .where(t.c.td_date.between(start_date, end_date))
    # return str(s.compile(compile_kwargs={'literal_binds':True})).replace('\n', '')
    return s

    # return "SELECT td_date FROM trade_dates WHERE td_date BETWEEN '%s' AND '%s'" % (start_date, end_date);

def raw_db_fund_value(start_date, end_date, codes=None, fund_ids=None, date_selector=None):

    # if not(isinstance(start_date, str)):
        # start_date = start_date.strftime("%Y-%m-%d")
    # if not(isinstance(end_date, str)):
        # start_date = end_date.strftime("%Y-%m-%d")

    if date_selector is None:
        raise NotImplementedError

    db = database.connection('base')
    t_fund = Table('ra_fund', MetaData(bind=db), autoload=True)
    t_nav = Table('ra_fund_nav', MetaData(bind=db), autoload=True).alias('A')
    date_selector = date_selector.alias('E')
    columns = [
            t_nav.c.ra_date.label('date'),
            t_nav.c.ra_code.label('code'),
            t_nav.c.ra_nav_adjusted
            ]
    if fund_ids is not None:
        #
        # 按照代码筛选基金
        #
        codes = [str(e) for e in fund_ids]
        code_selector = select([t_fund.c.globalid]) \
                .where(t_fund.c.globalid.in_(codes)) \
                .alias('D')
        # code_sql = "SELECT globalid FROM ra_fund WHERE globalid IN (%s)" % (code_str);
        s = select(columns) \
                .select_from(code_selector) \
                .select_from(date_selector) \
                .where(t_nav.c.ra_fund_id == code_selector.c.globalid) \
                .where(t_nav.c.ra_date == date_selector.c.td_date) \
                .order_by(t_nav.c.ra_date)
        # sql = "SELECT A.ra_date as date, A.ra_code as code, A.ra_nav_adjusted FROM ra_fund_nav A, (%s) D, (%s) E WHERE A.ra_fund_id = D.globalid AND A.ra_date = E.td_date ORDER BY A.ra_date" % (code_sql, date_sql)
    elif codes is not None:
        #
        # 按照代码筛选基金
        #
        codes = [str(e) for e in codes]
        code_selector = select([t_fund.c.globalid]) \
                .where(t_fund.c.ra_code.in_(codes)) \
                .alias('D')
        # code_sql = "SELECT globalid FROM ra_fund WHERE ra_code IN (%s)" % (code_str);
        s = select(columns) \
                .select_from(code_selector) \
                .select_from(date_selector) \
                .where(t_nav.c.ra_fund_id == code_selector.c.globalid) \
                .where(t_nav.c.ra_date == date_selector.c.td_date) \
                .order_by(t_nav.c.ra_date)
        # sql = "SELECT A.ra_date as date, A.ra_code as code, A.ra_nav_adjusted FROM ra_fund_nav A, (%s) D, (%s) E WHERE A.ra_fund_id = D.globalid AND A.ra_date = E.td_date ORDER BY A.ra_date" % (code_sql, date_sql)
    else:
        s = select(columns) \
                .select_from(date_selector) \
                .where(t_nav.c.ra_date == date_selector.c.td_date) \
                .order_by(t_nav.c.ra_date)
        # sql = "SELECT A.ra_date as date, A.ra_code as code, A.ra_nav_adjusted FROM ra_fund_nav A, (%s) E WHERE A.ra_date = E.td_date ORDER BY A.ra_date" % (date_sql)

    # logger.debug("db_fund_value: " + str(s.compile(compile_kwargs={'literal_binds':True})))

    df = pd.read_sql(s, db, index_col=['date', 'code'], parse_dates=['date'])
    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df

def db_fund_value(start_date, end_date, codes=None, fund_ids=None):
    #
    # 按照周收盘取净值
    #
    date_selector = build_sql_trade_date_weekly(start_date, end_date)
    return raw_db_fund_value(start_date, end_date, codes, fund_ids, date_selector)


def db_fund_value_daily(start_date, end_date, codes=None, fund_ids=None):
    #
    # 按照日收盘取净值
    #
    date_selector = build_sql_trade_date_daily(start_date, end_date)
    return raw_db_fund_value(start_date, end_date, codes, fund_ids, date_selector)

def raw_fund_value_by_type(start_date, end_date, date_selector, types, l2_type):
    '''
    types => List[yinhe_l2_types]
    l2_type => (stock, bond) => 1
               (money) => 2
    '''
    # if not(isinstance(start_date, str)):
        # start_date = start_date.strftime("%Y-%m-%d")
    # if not(isinstance(end_date, str)):
        # start_date = end_date.strftime("%Y-%m-%d")

    if date_selector is None or types is None:
        raise NotImplementedError

    #
    # [XXX] 本来想按照周收盘取净值数据, 但实践中发现周收盘存在美股和A
    # 股节假日对其的问题. 实践证明, 最好的方式是按照自然日的周五来对齐
    # 数据.
    #
    # 按照周收盘取净值
    #
    E = date_selector.alias('E')

    #
    # 按照基金类型筛选基金
    #
    db = database.connection('base')
    # if looking for stock / bond, type_sql is based on yinhe_type
    if l2_type == 1:
        t_yh = Table('yinhe_type', MetaData(bind=db), autoload=True)
        #type_sql
        B = select([func.distinct(t_yh.c.yt_fund_id).label('yt_fund_id')]) \
                .where(t_yh.c.yt_l2_type.in_(types)) \
                .where(t_yh.c.yt_begin_date <= end_date) \
                .where((t_yh.c.yt_end_date == '0000-00-00') | (t_yh.c.yt_end_date >= end_date)) \
                .alias('B')
        # type_sql = "SELECT DISTINCT yt_fund_id FROM yinhe_type WHERE (yt_l2_type IN ('200101', '200102', '200104', '200201', '200202')) AND (yt_begin_date <= '%s' AND (yt_end_date = '0000-00-00' OR yt_end_date >= '%s'))" % (end_date, end_date)
        #
    # ===================================================
    # Deprecated: There's no table `wind_fund_type`
    # elif l2_type == 2:
    # # if looking for money instead, type_sql is based on wind_fund_type
        # t_wind = Table('wind_fund_type', MetaData(bind=db), autoload=True)
        # B = select([func.distinct(t_wind.c.wf_fund_id).label('wf_fund_id')]) \
                # .where(t_wind.c.wf_type.like('20010104%%')) \
                # .where(t_wind.c.wf_start_time <= end_date) \
                # .where((t_wind.c.wf_end_time == None) | (t_wind.c.wf_end_time >= end_date))
    # type_sql = "SELECT DISTINCT wf_fund_id FROM wind_fund_type WHERE (wf_type like '20010104%%') AND (wf_start_time <= '%s' AND (wf_end_time IS NULL OR wf_end_time >= '%s'))" % (end_date, end_date);
    # ===================================================
    # 按照成立时间筛选基金
    #
    t_fund = Table('ra_fund', MetaData(bind=db), autoload=True)
    #regtime_sql
    C = select([func.distinct(t_fund.c.globalid).label('globalid')]) \
            .where(t_fund.c.ra_regtime <= start_date) \
            .where(t_fund.c.ra_regtime != '0000-00-00') \
            .where(t_fund.c.ra_mask.op('&')(0x01) == 0) \
            .alias('C')
    # regtime_sql = "SELECT DISTINCT globalid FROM ra_fund WHERE ra_regtime<='%s' and ra_regtime!='0000-00-00' AND (ra_mask & 0x01) = 0" % (start_date);
    #
    # 使用inner join 求交集
    #
    #intersected
    D = select([B.c.yt_fund_id]) \
            .select_from(join(B, C, B.c.yt_fund_id == C.c.globalid)) \
            .alias('D')
    # intersected = "SELECT B.yt_fund_id FROM (%s) AS B JOIN (%s) AS C ON B.yt_fund_id = C.globalid" % (type_sql, regtime_sql);
    #
    #
    t_nav = Table('ra_fund_nav', MetaData(bind=db), autoload=True).alias('A')
    columns = [
            t_nav.c.ra_date.label('date'),
            t_nav.c.ra_code.label('code'),
            t_nav.c.ra_nav_adjusted
            ]
    s = select(columns) \
            .select_from(D) \
            .select_from(E) \
            .where(t_nav.c.ra_fund_id == D.c.yt_fund_id) \
            .where(t_nav.c.ra_date == E.c.td_date) \
            .order_by(t_nav.c.ra_date)
    # sql = "SELECT A.ra_date as date, A.ra_code as code, A.ra_nav_adjusted FROM ra_fund_nav A, (%s) D, (%s) E WHERE A.ra_fund_id = D.yt_fund_id AND A.ra_date = E.td_date ORDER BY A.ra_date" % (intersected, date_sql);



    # logger.debug("raw_fund_value: " + str(s.compile(compile_kwargs={'literal_binds':True})))

    df = pd.read_sql(s, db, index_col = ['date', 'code'], parse_dates=['date'])

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df

def stock_fund_value(start_date, end_date):
    types = ['200101', '200102', '200104', '200201', '200202']
    date_selector = build_sql_trade_date_weekly(start_date, end_date)
    return raw_fund_value_by_type(start_date, end_date, date_selector, types, 1)

def stock_day_fund_value(start_date, end_date):
    types = ['200101', '200102', '200104', '200201', '200202']
    date_selector = build_sql_trade_date_daily(start_date, end_date)
    return raw_fund_value_by_type(start_date, end_date, date_selector, types, 1)



def bond_fund_value(start_date, end_date):
    types = ['200301']
    date_selector = build_sql_trade_date_weekly(start_date, end_date)
    return raw_fund_value_by_type(start_date, end_date, date_selector, types, 1)

def bond_day_fund_value(start_date, end_date):
    types = ['200301']
    date_selector = build_sql_trade_date_daily(start_date, end_date)
    return raw_fund_value_by_type(start_date, end_date, date_selector, types, 1)



# Deprecated: There's no table `wind_fund_type`
def money_fund_value(start_date, end_date):
    date_selector = build_sql_trade_date_weekly(start_date, end_date)
    return raw_fund_value_by_type(start_date, end_date, date_selector, [0], 2)

# Deprecated: There's no table `wind_fund_type`
def money_day_fund_value(start_date, end_date):
    date_selector = build_sql_trade_date_daily(start_date, end_date)
    return raw_fund_value_by_type(start_date, end_date, date_selector, [0], 2)


def raw_db_index_value(start_date, end_date, codes=None, date_selector=None):
    #
    # 按照周收盘取净值
    #

    # if not(isinstance(start_date, str)):
        # start_date = start_date.strftime("%Y-%m-%d")
    # if not(isinstance(end_date, str)):
        # start_date = end_date.strftime("%Y-%m-%d")

    if date_selector is None:
        raise NotImplementedError

    db = database.connection('base')
    t_idx = Table('ra_index', MetaData(bind=db), autoload=True)
    A = Table('ra_index_nav', MetaData(bind=db), autoload=True).alias('A')
    E = date_selector.alias('E')
    columns = [
            A.c.ra_date.label('date'),
            A.c.ra_index_code.label('code'),
            A.c.ra_nav
            ]
    if codes is not None:
        #
        # 按照代码筛选基金
        #
        codes = [str(e) for e in codes]

        #code_selector
        D = select([t_idx.c.globalid]) \
                .where(t_idx.ra_code.in_(codes)) \
                .alias(D)
        #date_selector

        # code_sql = "SELECT globalid FROM ra_index WHERE ra_code IN (%s)" % (code_str);
        s = select(columns) \
                .select_from(D) \
                .select_from(E) \
                .where(A.c.ra_index_id == D.c.globalid) \
                .where(A.c.ra_date == E.c.td_date) \
                .order_by(A.c.ra_date)

        # sql = "SELECT A.ra_date as date, A.ra_index_code as code, A.ra_nav FROM ra_index_nav A, (%s) D, (%s) E WHERE A.ra_index_id = D.globalid AND A.ra_date = E.td_date ORDER BY A.ra_date" % (code_sql, date_sql)
    else:
        s = select(columns) \
                .select_from(E) \
                .where(A.c.ra_date == E.c.td_date) \
                .order_by(A.c.ra_date)
        # sql = "SELECT ra_date as date, ra_index_code, ra_nav FROM ra_index_nav, (%s) E WHERE ra_date = E.td_date ORDER BY ra_date" % (date_sql)

    # logger.debug("raw_index_value: " + str(s.compile(compile_kwargs={'literal_binds':True})))

    df = pd.read_sql(s, db, index_col = ['date', 'code'], parse_dates=['date'])
    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df

def db_index_value(start_date, end_date, codes=None):
    date_selector = build_sql_trade_date_weekly(start_date ,end_date)
    return raw_db_index_value(start_date, end_date, codes=codes, date_selector=date_selector)

def db_index_value_daily(start_date, end_date, codes=None):
    date_selector = build_sql_trade_date_daily(start_date ,end_date)
    return raw_db_index_value(start_date, end_date, codes=codes, date_selector=date_selector)



def index_value(start_date, end_date):
    date_selector = build_sql_trade_date_weekly(start_date, end_date)
    return raw_db_index_value(start_date, end_date, codes=None, date_selector=date_selector)


def index_day_value(start_date, end_date):
    date_selector = build_sql_trade_date_daily(start_date, end_date)
    return raw_db_index_value(start_date, end_date, codes=None, date_selector=date_selector)



def other_fund_value(start_date, end_date):
    df = index_value(start_date, end_date)
    return df[[Const.sp500_code, Const.gold_code, Const.hs_code]]

def other_day_fund_value(start_date, end_date):
    df = index_day_value(start_date, end_date)
    return df[[Const.sp500_code, Const.gold_code, Const.hs_code]]


# Deprecated: There's no table `wind_fund_stock_value`
# def position():


    # #dates = set()

    # conn  = MySQLdb.connect(**config.db_base)
    # cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    # conn.autocommit(True)


    # sql = "select wf_fund_code, wf_time, wf_stock_value from wind_fund_stock_value"

    # cur.execute(sql)
    # records = cur.fetchall()

    # position_dict = {}

    # code_position = {}


    # for record in records:

        # code      = record['wf_fund_code']
        # position  = record['wf_stock_value']
        # ps        = code_position.setdefault(code, [])
        # ps.append(position)

    # conn.close()

    # for code in code_position.keys():
        # position_dict[code] = np.mean(code_position[code])


    # dates = list(dates)
    # dates.sort()

    # position_values = []
    # position_codes  = []
    # for code in position_dict.keys():
        # position_codes.append(code)
        # ps = []
        # ps_dict = position_dict[code]
        # for d in dates:
            # if ps_dict.has_key(d):
                # ps.append(ps_dict[d])
            # else:
                # ps.append(np.NaN)


        # position_values.append(ps)
        # #print code , len(vs)


    # df = pd.DataFrame(np.matrix(position_values).T, index = dates, columns = position_codes)
    # df = df.fillna(method='pad')
    # df.index.name = 'date'
    # return df



# Deprecated: There's no table `wind_fund_type`
# def scale():


    # conn  = MySQLdb.connect(**config.db_base)
    # cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    # conn.autocommit(True)


    # sql = "select a.fi_code,a.fi_laste_raise from fund_infos a inner join wind_fund_type b on a.fi_globalid=b.wf_fund_id  where b.wf_flag=1 and (b.wf_type like '20010101%%' or b.wf_type like '2001010201%%' or b.wf_type like '2001010202%%' or b.wf_type like '2001010204%%'  )"


    # df = pd.read_sql(sql, conn, index_col = 'fi_code')

    # #cur.execute(sql)

    # #records = cur.fetchall()
    # conn.close()

    # return df



if __name__ == '__main__':
    pass

    # sql = "SELECT ra_date, ra_code, ra_nav_adjusted FROM `ra_fund_nav` JOIN trade_dates ON ra_date = td_date WHERE ra_code IN ('206018', '100058', '000509') AND ra_date >= '2015-10-20' ORDER BY ra_date"

    # print "stock_fund_value", sql
    
    # conn  = MySQLdb.connect(**config.db_base)
    # df = pd.read_sql(sql, conn, index_col = ['ra_date', 'ra_code'], parse_dates=['ra_date'])
    # conn.close()

    # df = df.unstack().fillna(method='pad')
    # df.columns = df.columns.droplevel(0)
    
    # df.to_csv("~/tmp.csv", columns=['206018', '100058', '000509'])
    #df = stock_fund_value('2014-01-03', '2016-06-03')
    #df = bond_fund_value('2014-01-03', '2016-06-03')
    #df = money_fund_value('2015-01-03', '2016-06-03')
    #df =  index_value('2010-01-28', '2016-06-03')
    #f =  other_fund_value('2014-01-03', '2016-06-03')
    #df =  position()
    #df =  scale()
    #df  = bond_day_fund_value('2014-01-03', '2016-06-03')
    #df  = bond_fund_value('2014-01-01', '2016-07-19')
    #df.to_csv(datapath('bond.csv'))
    #print all_trade_dates()
    #print trade_date_index('2014-01-03', '2016-06-03')
