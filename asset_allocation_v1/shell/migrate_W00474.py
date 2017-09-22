# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
from db import database
from sqlalchemy import MetaData, Table, select

def load_caihui():
    db = database.connection('caihui')
    metadata = MetaData(bind = db)
    t = Table('tq_qt_index', metadata, autoload = True)

    columns = [
            t.c.TRADEDATE.label('date'),
            t.c.TOPEN,
            t.c.TCLOSE,
            t.c.THIGH,
            t.c.TLOW,
            t.c.VOL,
            t.c.AMOUNT,
            ]

    s = select(columns)\
            .where(t.c.TRADEDATE.between('2005-01-01', '2020-01-01'))\
            .where(t.c.SECODE == 2070006523)\
            .order_by(t.c.TRADEDATE)

    df = pd.read_sql(s, db, index_col = ['date'], parse_dates = ['date'])
    #print df
    return df


def insert_base(df):
    db = database.connection('base')

    df.columns = ['ra_open', 'ra_nav', 'ra_high', 'ra_low', 'ra_volume'\
            ,'ra_amount']
    df['ra_nav_date'] = df.index
    df = df.reindex(pd.date_range(start = df.index[0].date(), end = df.index[-1].date(),\
            freq = 'D'))
    df['ra_mask'] = np.sign(df['ra_open'])
    df['ra_mask'].fillna(0, inplace = True)
    df['ra_mask'] = df['ra_mask'].astype('int')
    df['ra_mask'] = 1 - df['ra_mask']

    df.fillna(method = 'ffill', inplace = True)

    df['ra_index_id'] = 120000041
    df['ra_index_code'] = 'W00474'
    df['ra_inc'] = df.ra_nav.pct_change()
    df.fillna(0, inplace = True)
    df['ra_date'] = df.index
    df['created_at'] = datetime.datetime.now()
    df['updated_at'] = datetime.datetime.now()

    df.to_sql("ra_index_nav", db, if_exists='append', index=False)

    df2 = pd.DataFrame({'globalid':[120000042], 'ra_code':['W00475'], 'ra_caihui_code':\
            [2070006523], 'ra_name':['标普高盛黄金'], 'ra_announce_date':[\
            datetime.datetime(2000,1,1)], 'ra_begin_date':[datetime.datetime(2000,1,1)],\
            'ra_base_date':[datetime.datetime(2000,1,1)],'created_at':[datetime.datetime.now()],\
            'updated_at':[datetime.datetime.now()]})

    df2.to_sql("ra_index", db, if_exists='append', index=False)
    #print df


if __name__ == '__main__':
    df = load_caihui()
    #df.to_csv('W00474.csv', index = False)
    #df = pd.read_csv('W00474.csv', index_col = ['date'], parse_dates = ['date'])
    insert_base(df)
