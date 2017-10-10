#coding=utf8

from sqlalchemy import MetaData, Table, select
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database


logger = logging.getLogger(__name__)

#
# wt_filter
#
def load(filters):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('wt_filter', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.wt_name,
        t1.c.wt_filter_num,
        t1.c.wt_index_id,
        t1.c.wt_begin_date,
    ]

    s = select(columns)
    if filters is not None:
        s = s.where(t1.c.globalid.in_(filters))

    df = pd.read_sql(s, db)
    return df

