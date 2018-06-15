#coding=utf-8
import pandas as pd
import numpy as np
import warnings
from asset import Asset
from sqlalchemy import *
from db import *
from trade_date import *
import weakref

class BondIndex(Asset):
    __cache = weakref.WeakValueDictionary()

    def __new__(cls, secode, *args, **kwargs):
        secode = str(secode)
        if secode in cls.__cache:
            return cls.__cache[secode]
        return super(BondIndex, cls).__new__(cls)

    def __init__(self, secode):
        secode = str(secode)
        if secode in self.__cache:
            return
        name = load_bond_index_info(secode).loc[secode, "INDEXNAME"]
        nav_sr = load_cbdindex_nav(secode)
        #若未取到, 尝试从tq_qt_index中读
        if nav_sr.empty:
            nav_sr = load_index_nav(secode)
        super(BondIndex, self).__init__(secode, name=name, nav_sr=nav_sr)
        self.__cache[secode] = self

    def nav(self, begin_date = None, end_date = None, reindex = None):
        if begin_date is None:
            begin_date = '2010-01-01'
        if end_date is None:
            end_date = '2018-05-01'
        if reindex is None:
            reindex = ATradeDate.week_trade_date()
        nav = super(BondIndex, self).nav(reindex=reindex).dropna()
        nav = nav[(begin_date <= nav.index) & (nav.index <= end_date)]
        nav.name = self.name
        #  nav = nav.loc[begin_date:end_date]
        if nav.empty:
            return pd.Series()
        return nav/nav[0]

    def inc(self, begin_date = None, end_date = None, reindex = None):
        return self.nav(begin_date, end_date, reindex).pct_change().fillna(0)


#读取数据库
def load_bond_index_info(secode = None):
    db = database.connection('caihui')
    t1 = Table('tq_ix_basicinfo', MetaData(bind=db), autoload=True)
    columns = [t1.c.SECODE, t1.c.INDEXNAME]
    if secode == None:
        s = select(columns).where((t1.c.INDEXTYPE == 4) and t1.c.ISVALID)
    else:
        s = select(columns).where(t1.c.SECODE == secode)
    df = pd.read_sql(s, db, index_col="SECODE")
    return df

def load_cbdindex_nav(secode):
    db = database.connection('caihui')
    t1 = Table('tq_qt_cbdindex', MetaData(bind=db), autoload=True)
    columns = [t1.c.TRADEDATE, t1.c.TCLOSE, t1.c.DIRTYCLOSE, t1.c.CHANGEPTC]
    s = select(columns).where(t1.c.SECODE == secode)
    df = pd.read_sql(s, db, index_col='TRADEDATE', parse_dates=['TRADEDATE'])
    if (df.CHANGEPTC.fillna(0) == 0).all():
        return df.DIRTYCLOSE
    else:
        return df.TCLOSE
    #  return df.TCLOSE

def load_cbdindex_dur_cvx(secode, reindex=None):
    db = database.connection('caihui')
    t1 = Table('tq_qt_cbdindex', MetaData(bind=db), autoload=True)
    columns = [t1.c.TRADEDATE, t1.c.AVGMVDURATION, t1.c.AVGMKCONVEXITY]
    s = select(columns).where(t1.c.SECODE == secode)
    df = pd.read_sql(s, db, index_col='TRADEDATE', parse_dates=['TRADEDATE'])
    if reindex is not None:
        df = df.reindex(reindex)
    return df.AVGMVDURATION, df.AVGMKCONVEXITY


#  def load_cbdindex_inc(secode):
    #  db = database.connection('caihui')
    #  t1 = Table('tq_qt_cbdindex', MetaData(bind=db), autoload=True)
    #  columns = [t1.c.TRADEDATE, t1.c.CHANGEPTC]
    #  s = select(columns).where(t1.c.SECODE == secode)
    #  df = pd.read_sql(s, db, index_col='TRADEDATE', parse_dates=['TRADEDATE'])
    #  return df.squeeze()


def load_index_nav(secode):
    db = database.connection('caihui')
    t1 = Table('tq_qt_index', MetaData(bind=db), autoload=True)
    columns = [t1.c.TRADEDATE, t1.c.TCLOSE]
    s = select(columns).where(t1.c.SECODE == secode)
    df = pd.read_sql(s, db, index_col='TRADEDATE', parse_dates=['TRADEDATE'])
    return df.squeeze()

#  def get_fund_nav(gid, begin_date = '2010-01-01', end_date='2018-05-01'):
    #  if os.path.exists("tmpfund/%s.csv" % gid):
        #  nav = pd.read_csv("tmpfund/%s.csv" % gid, index_col=0, parse_dates=True, header=None).squeeze()
        #  nav.index.name = "date"
        #  nav.name = "nav"
    #  else:
        #  nav = base_ra_fund_nav.load_series(gid)
        #  if not nav.empty:
            #  nav.to_csv("tmpfund/%s.csv" % gid)
        #  else:
            #  print gid
    #  return nav.reindex(tdate).loc[begin_date:end_date]

#  def get_fund_inc(gid, begin_date = '2010-01-01', end_date='2018-05-01'):
    #  return get_fund_nav(gid, begin_date, end_date).pct_change().fillna(0)
