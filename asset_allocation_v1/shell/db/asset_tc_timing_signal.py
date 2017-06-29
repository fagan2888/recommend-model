#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

# #
# # mz_reshape
# #
# def load(gids, xtypes=None):
#     db = database.connection('asset')
#     metadata = MetaData(bind=db)
#     t1 = Table('rs_reshape', metadata, autoload=True)

#     columns = [
#         t1.c.globalid,
#         t1.c.rs_type,
#         t1.c.rs_pool,
#         t1.c.rs_asset,
#         t1.c.rs_name,
#     ]

#     s = select(columns)

#     if gids is not None:
#         s = s.where(t1.c.globalid.in_(gids))
#     if xtypes is not None:
#         s = s.where(t1.c.rs_type.in_(xtypes))
    
#     df = pd.read_sql(s, db)

#     return df
def load(timings):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('tc_timing_signal', metadata, autoload=True)

    columns = [
        t1.c.tc_timing_id,
        t1.c.tc_date,
        t1.c.tc_signal,
    ]

    s = select(columns)

    if timings is not None:
        if hasattr(timings, "__iter__") and not isinstance(timings, str):
            s = s.where(t1.c.tc_timing_id.in_(timings))
        else:
            s = s.where(t1.c.tc_timing_id == timings)
    
    df = pd.read_sql(s, db, index_col = ['tc_date', 'tc_timing_id'], parse_dates=['tc_date'])

    df = df.unstack().fillna(method='pad')
    df.columns = df.columns.droplevel(0)

    return df


def load_series(timing_id):
    # 加载基金列表
    db = database.connection('asset')
    t = Table('tc_timing_signal', MetaData(bind=db), autoload=True)
    columns = [
        t.c.tc_date,
        t.c.tc_signal
    ]
    s = select(columns, (t.c.tc_timing_id == timing_id))

    df = pd.read_sql(s, db, index_col = ['tc_date'], parse_dates=['tc_date'])

    return df['tc_signal']

