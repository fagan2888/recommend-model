#coding=utf-8
'''
Created at Feb. 21, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import sys
import logging
from sqlalchemy import MetaData, Table, select, func, and_, not_
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta
from ipdb import set_trace
sys.path.append('shell')
from db import database
from db import base_ra_index

logger = logging.getLogger(__name__)

def cal_bfp_nav(index_id, ret):

    dates = pd.date_range('2009-12-31', '2019-02-21')
    columns = pd.Index(['ra_index_id', 'ra_index_code', 'ra_nav', 'ra_inc', 'ra_nav_date'])
    df = pd.DataFrame(index=dates, columns=columns)
    df.index.name = 'ra_date'

    df['ra_index_id'] = index_id
    df['ra_index_code'] = base_ra_index.find(index_id)[1]
    df['ra_nav_date'] = df.index

    ra_nav = 1000.0
    ret = (1 + ret) ** (1 / 365) - 1
    for date in dates:
        df.loc[date, 'ra_nav'] = round(ra_nav, 4)
        ra_nav = ra_nav * (1 + ret)
    df.loc[:, 'ra_inc'] = round(df['ra_nav'].pct_change().iloc[1:], 8)

    df = df.reset_index().set_index(['ra_index_id', 'ra_date'])

    return df

def cal_all_bfp_nav():

    df_bfp_1w = cal_bfp_nav('120000095', 0.036)
    df_bfp_1m = cal_bfp_nav('120000096', 0.038)
    df_bfp_2m = cal_bfp_nav('120000097', 0.040)
    df_bfp_3m = cal_bfp_nav('120000098', 0.041)
    df_bfp_6m = cal_bfp_nav('120000099', 0.042)
    df_bfp_1y = cal_bfp_nav('120000100', 0.042)

    df = pd.concat([df_bfp_1w, df_bfp_1m, df_bfp_2m, df_bfp_3m, df_bfp_6m, df_bfp_1y])

    return df


def load_ra_index_nav_data(index_ids):

    engine = database.connection('base')
    metadata = MetaData(bind=engine)
    t = Table('ra_index_nav', metadata, autoload=True)

    columns = [
        t.c.ra_index_id,
        t.c.ra_index_code,
        t.c.ra_date,
        t.c.ra_nav,
        t.c.ra_inc,
        t.c.ra_nav_date
    ]

    s = select(columns)
    s = s.where(t.c.ra_index_id.in_(index_ids))

    df = pd.read_sql(s, engine, index_col=['ra_index_id', 'ra_date'], parse_dates=['ra_date', 'ra_nav_date'])

    return df

def insert_ra_index_nav(df_new, begin_date=None, end_date=None, fund_codes=None):

    index_ids = ['120000095','120000096','120000097','120000098','120000099','120000100']
    df_old = load_ra_index_nav_data(index_ids)

    db = database.connection('base')
    t = Table('ra_index_nav', MetaData(bind=db), autoload=True)
    database.batch(db, t, df_new, df_old)


if __name__=='__main__':

    # cal_index_nav('120000095', 0.04)
    # cal_all_bfp_nav()
    insert_ra_index_nav(cal_all_bfp_nav())
