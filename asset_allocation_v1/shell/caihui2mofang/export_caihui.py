#coding=utf8


import sys
sys.path.append('./shell')
from sqlalchemy import *
import logging
import config
import pandas as pd
from db import database


logger = logging.getLogger(__name__)


def export_caihui_table(table, columns, start_date = None, end_date = None, index_col = None):

    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table(table, metadata, autoload=True)
    s = select(column).select_from(t)

    if start_date is not None:
        s = s.where(t.c.ENTRYDATE >= start_date)
    if end_date is not None:
        s = s.where(t.c.ENTRYDATE <= end_date)

    df = pd.read_sql(s, db, index_col = index_col)

    return df


def export_caihui(schedule_df):
    print schedule_df['table']
    return 0


def update_caihui(schedule):
    return 0


if __name__ == '__main__':

    schedule_df = pd.read_csv('data/export_caihui_data.csv', index_col = ['table'])
    export_caihui(schedule_df)
    #df = export_caihui('tq_sk_basicinfo',['secode', 'symbol'], start_date = '2017-08-01', end_date = '2017-08-18', index_col = ['secode'])
