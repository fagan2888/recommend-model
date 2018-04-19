from sqlalchemy import MetaData, Table, select, func, literal_column
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database

logger = logging.getLogger(__name__)

def load_series(_id):
    db = database.connection('asset')
    t1 = Table('rm_riskmgr_mgarch_signal', MetaData(bind=db), autoload=True)
    
    columns = [
        t1.c.rm_date,
        t1.c.target_rm_riskmgr_id,
        t1.c.rm_signal
    ]

    index_col = ['rm_date', 'target_rm_riskmgr_id']
    
    s = select(columns)

    if _id is not None:
        s = s.where(t1.c.rm_riskmgr_id == _id)
    else:
        return None
    
    df = pd.read_sql(s, db, index_col=index_col, parse_dates = ['rm_date']).unstack()
    df.columns = df.columns.levels[1]
    return df


def save(_id, df):
    db = database.connection('asset')
    t2 = Table('rm_riskmgr_mgarch_signal', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns, (t2.c.rm_riskmgr_id == _id))
    df_old = pd.read_sql(s, db, index_col = ['rm_riskmgr_id', 'rm_date', 'target_rm_riskmgr_id'], parse_dates=['rm_date'])
    database.batch(db, t2, df, df_old, timestamp=True)