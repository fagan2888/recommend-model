#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
import numpy as np
# import os
# import sys
import logging
import database
import datetime

from dateutil.parser import parse
from ipdb import set_trace
import trade_date
import re

logger = logging.getLogger(__name__)

def get_secode(ra_code):
    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table('tq_fd_basicinfo', metadata, autoload=True)

    columns = [
        t.c.FSYMBOL.label('ra_code'),
        t.c.SECURITYID.label('secode'),
    ]

    s = select(columns).where(t.c.FSYMBOL.in_(ra_code))
    df = pd.read_sql(s, db, index_col = ['ra_code'])

    return df


def get_tshare(secode, reindex = None):
    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table('tq_fd_fshare', metadata, autoload=True)

    columns = [
        t.c.TOTALSHARE.label('tshare'),
        t.c.DECLAREDATE.label('date'),
        t.c.SECURITYID.label('secode'),
    ]

    s = select(columns).where(t.c.SECURITYID.in_(secode))
    df = pd.read_sql(s, db, parse_dates = ['date'])
    df = df.groupby(['date', 'secode']).last()
    df = df.unstack().fillna(method = 'pad')
    df.columns = df.columns.get_level_values(1)

    if reindex is not None:
        df = df.reindex(reindex).fillna(method = 'pad')

    return df


def get_iratio(secode, reindex = None):
    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table('tq_fd_sharestat', metadata, autoload=True)

    columns = [
        t.c.INVTOTRTO.label('iratio'),
        t.c.PUBLISHDATE.label('date'),
        t.c.SECODE.label('secode'),
    ]

    s = select(columns).where(t.c.SECODE.in_(secode))
    df = pd.read_sql(s, db, parse_dates = ['date'])
    df = df.groupby(['date', 'secode']).last()
    df = df.unstack().fillna(method = 'pad')
    df.columns = df.columns.get_level_values(1)

    if reindex is not None:
        df = df.reindex(reindex).fillna(method = 'pad')

    return df


def get_totyears(secode, reindex = None):

    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t1 = Table('tq_fd_mgperformance', metadata, autoload=True)

    columns = [
        t1.c.MANAGERCODE.label('pscode'),
        t1.c.SECODE.label('secode'),
        t1.c.BEGINDATE.label('sdate'),
        t1.c.ENDDATE.label('edate'),
    ]

    dict_totyears = {}
    s = select(columns).where(t1.c.SECODE.in_(secode))
    df = pd.read_sql(s, db, parse_dates = ['sdate', 'edate'])
    df.edate = df.edate.replace(datetime.datetime(1900,1,1),datetime.datetime(2100,1,1))
    for code in secode:
        dict_totyears[code] = {}
        for date in reindex:
            tmp_df = df[(df.secode == code) & (df.sdate < date) & (df.edate > date)]
            if len(tmp_df) == 0:
                dict_totyears[code][date] = np.nan
            else:
                mdays = []
                for pscode in tmp_df.pscode.values:
                    mdays.append(get_mdays(pscode, date))
                dict_totyears[code][date] = np.mean(mdays)

            # print pd.DataFrame((dict_totyears))
        df_totyears = pd.DataFrame(dict_totyears)
        if not len(df_totyears.columns) % 10:
            df_totyears.to_csv('data/m_totyears2.csv', index_label = 'date')

    return df_totyears


def get_mdays(pscode, date):

    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t1 = Table('tq_fd_managersta', metadata, autoload=True)

    columns = [
        t1.c.PSCODE.label('pscode'),
        t1.c.TOTYEARS.label('totyears'),
    ]

    s = select(columns).where(t1.c.PSCODE == pscode)
    df = pd.read_sql(s, db)
    if len(df) == 0:
        return np.nan

    else:
        years_list = re.findall(r'(\w*[0-9]+)\w*',df.totyears.loc[0])
        years_list = [int(x) for x in years_list]
        if len(years_list) == 1:
            years_list = [0] + years_list
        days = years_list[0]*365 + years_list[1]
        days_tillnow = (datetime.datetime.now()-date).days

        return days - days_tillnow




if __name__ == '__main__':

    dates = trade_date.ATradeDate.trade_date('2010-01-01')
    # df = get_securityid(['519983'])
    # df = get_iratio(['1030000867', '1030000868'])
    # df = get_totyears(['1030000867', '1030000868'], dates)
    print df
