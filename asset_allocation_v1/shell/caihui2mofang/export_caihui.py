#coding=utf8


import sys
sys.path.append('./shell')
from sqlalchemy import *
import logging
import config
import pandas as pd
import numpy as np
from db import database


logger = logging.getLogger(__name__)


def save(df, table):
   df.to_csv('./caihui_table/%s.csv' % table)


def export_caihui_table(table, columns, start_date = None, end_date = None, index_col = None):

    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table(table, metadata, autoload=True)
    s = select(columns).select_from(t)

    if start_date is not None and not np.isnan(start_date):
        s = s.where(t.c.ENTRYDATE >= start_date)
    if end_date is not None and not np.isnan(end_date):
        s = s.where(t.c.ENTRYDATE <= end_date)

    df = pd.read_sql(s, db, index_col = index_col)

    return df


def export_caihui(schedule_df):
    for table in schedule_df.index:
        if 1 == schedule_df.loc[table, 'export']:
            df = export_caihui_table(
                    table,
                    [c for c in schedule_df.loc[table, 'column'].split()],
                    start_date = schedule_df.loc[table, 'start_date'],
                    end_date = schedule_df.loc[table, 'end_date'],
                    index_col = schedule_df.loc[table, 'index_col']
            )
            print df
            print 'export %s done' % table
            save(df, table)


def update_caihui(schedule):
    return 0


if __name__ == '__main__':

    schedule_df = pd.read_csv('data/export_caihui_data.csv', index_col = ['table'])
    export_caihui(schedule_df)
    #df = export_caihui_table('tq_sk_basicinfo',['secode', 'symbol'], start_date = '2017-08-01', end_date = '2017-08-18', index_col = ['secode'])
