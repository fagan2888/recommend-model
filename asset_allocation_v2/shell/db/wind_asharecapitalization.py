#coding=utf-8
'''
Created on: May. 14, 2019
Modified on: Jun. 3, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database
from . import util_db


logger = logging.getLogger(__name__)


def load_a_stock_total_share(stock_ids=None):

    stock_ids = util_db.to_list(stock_ids)

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareCapitalization', metadata, autoload=True)

    columns = [
        t.c.S_INFO_WINDCODE.label('stock_id'),
        t.c.CHANGE_DT.label('change_date'),
        t.c.CHANGE_DT1.label('change_date1'),
        t.c.ANN_DT.label('ann_date'),
        t.c.TOT_SHR.label('total_share')
    ]

    s = select(columns)
    if stock_ids is not None:
        s = s.where(t.c.S_INFO_WINDCODE.in_(stock_ids))

    df = pd.read_sql(s, engine, parse_dates=['change_date', 'change_date1', 'ann_date'])
    df['begin_date'] = df[['change_date', 'ann_date']].apply(max, axis='columns')
    df.sort_values(
        by=['stock_id', 'begin_date', 'change_date'],
        ascending=[True, True, False],
        inplace=True
    )
    df.drop(
        df.loc[
            (df.stock_id==df.shift(1).stock_id) & \
            (df.change_date<=df.shift(1).change_date)
        ].index,
        inplace=True
    )
    df.set_index(['stock_id', 'begin_date'], inplace=True)
    df.drop(['change_date', 'change_date1', 'ann_date'], axis='columns', inplace=True)

    # ix = pd.date_range(
        # start=df.index.levels[1][0],
        # end=df.index.levels[1][-1],
        # freq='D'
    # )
    # df = df.total_share.unstack().T.reindex(ix)

    # if fill_method is not None:
        # df.fillna(method=fill_method, inplace=True)
    # if begin_date is not None:
        # df = df.loc[begin_date:]
    # if end_date is not None:
        # df = df.loc[:end_date]
    # if reindex is not None:
        # df = df.reindex(reindex)

    return df


if __name__ == '__main__':

    load_a_stock_total_share('601598.SH')
    load_a_stock_total_share(begin_date='2019-01-01', end_date='2019-04-30')

