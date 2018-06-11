import pandas as pd
import numpy as np
from db import *
from sqlalchemy import *

def load_data(gid):
    db = database.connection('asset')
    t1 = Table('cp_competitor_nav', MetaData(bind=db), autoload=True)
    columns = [t1.c.cp_date, t1.c.cp_asset, t1.c.cp_nav, t1.c.cp_inc]
    s = select(columns).where(t1.c.cp_competitor_id == gid)
    df = pd.read_sql(s, db, index_col = 'cp_date', parse_dates=True)
    return df

def save(df):
    fmt_columns = ['cp_asset', 'cp_nav', 'cp_inc']
    fmt_precision = 6
    if not df.empty:
        df = database.number_format(df, fmt_columns, fmt_precision)
    db = database.connection('asset')
    t2 = Table('cp_competitor_nav', MetaData(bind=db), autoload=True)
    columns = [t2.c.cp_competitor_id, t2.c.cp_date, t2.c.cp_asset, t2.c.cp_nav, t2.c.cp_inc]
    s = select(columns)
    df_old = pd.read_sql(s, db, index_col = ['cp_competitor_id', 'cp_date'], parse_dates=['cp_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, fmt_columns, fmt_precision)
    database.batch(db, t2, df, df_old, timestamp=False)

def get_df_from_excel(path):
    df_raw = pd.read_excel(path, index_col='cp_date', parse_dates=True)
    nav_raw = df_raw.drop('cp_name')
    nav_raw = nav_raw.infer_objects().rename(index=lambda x: pd.to_datetime(x)-pd.Timedelta('1d'))
    dates = pd.date_range(nav_raw.index[0], nav_raw.index[-1])
    #  nav_raw = nav_raw.reindex(dates).fillna(method='pad')
    nav_raw = nav_raw.reindex(dates).interpolate()
    inc_raw = nav_raw.pct_change().fillna(0)
    inc_tmp = np.log(nav_raw).diff().fillna(0)
    nav = np.exp(inc_tmp.cumsum())
    df = pd.DataFrame({'cp_asset':nav_raw.stack(), 'cp_inc':inc_raw.stack(), 'cp_nav': nav.stack()})
    df.index.names = ['cp_date', 'cp_competitor_id']
    df = df.swaplevel(0,1).sort_index(level=0)
    return df
