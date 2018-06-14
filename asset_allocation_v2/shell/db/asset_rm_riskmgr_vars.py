from sqlalchemy import MetaData, Table, select, func, literal_column
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
from . import database

logger = logging.getLogger(__name__)

def load_series(_id):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('rm_riskmgr_vars', metadata, autoload=True)
    
    columns = [
        t1.c.ix_date,
        t1.c.var_2d,
        t1.c.var_3d,
        t1.c.var_5d
    ]

    index_col = ['ix_date']
    
    s = select(columns)

    if _id is not None:
        s = s.where(t1.c.index_id == _id)
    else:
        return None
    
    df = pd.read_sql(s, db, index_col=index_col, parse_dates = ['ix_date'])
    return df


def save(_id, df):
    fmt_columns = ['var_2d', 'var_3d', 'var_5d']
    fmt_precision = 6
    if not df.empty:
        df = database.number_format(df, fmt_columns, fmt_precision)
    
    db = database.connection('asset')
    t2 = Table('rm_riskmgr_vars', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns, (t2.c.index_id == _id))
    df_old = pd.read_sql(s, db, index_col = ['index_id', 'ix_date'], parse_dates=['ix_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, fmt_columns, fmt_precision)
    database.batch(db, t2, df, df_old, timestamp=True)