#coding=utf8

from sqlalchemy import MetaData, Table, select, func
import pandas as pd
import datetime
import logging
import database
import os
import numpy as np
from dateutil.parser import parse
logger = logging.getLogger(__name__)
def get_asset_newest_view(viewid):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('vw_view_inc', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # qry = session.query(
    #     func.max(t.vw_date).label("newest_date"), \
    # )
    # metadata = MetaData(bind=db)
    # t = Table('vw_view_inc', metadata, autoload=True)
    columns = [
        func.max(t.c.vw_date).label('newest_date'),
    ]
    s = select(columns).where(t.c.vw_view_id == viewid)
    df = pd.read_sql(s, db)
    return df
def insert_predict_pct(view_df):
    if type(view_df) != pd.core.frame.DataFrame:
        return (2, "view_df is not DataFrame")
    if view_df.empty:
        return (3, "view_df is null")
    try:
        db = database.connection('asset')
        ids = view_df['ids']
        dates = view_df['dates']
        inc = view_df['means']
        create_at = view_df['create_time']
        update_at = view_df['update_time']
        d = {
                'vw_view_id':ids,
                'vw_date':dates,
                'vw_inc':inc,
                'created_at':create_at,
                'updated_at':update_at,
        }
        test_df = pd.DataFrame(data=d)
        result = (0, "Insert into vw_veiw_inc success")
        df = test_df.to_sql("vw_view_inc", db, if_exists='append', index=False)
    except Exception, e:
        result = (1, "Insert into vw_veiw_inc fail: " + e.message)
    return result

def get_asset_day_view(viewid, day):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('vw_view_inc', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # qry = session.query(
    #     func.max(t.vw_date).label("newest_date"), \
    # )
    # metadata = MetaData(bind=db)
    # t = Table('vw_view_inc', metadata, autoload=True)
    columns = [
        #func.max(t.c.vw_date).label('newest_date'),
        t.c.vw_inc.label('inc')
    ]
    s = select(columns).where(t.c.vw_view_id == viewid).where(t.c.vw_date == day)
    df = pd.read_sql(s, db)
    return df
if __name__ == "__main__":
    assets = {
                '42110102':'means_000300.csv',
                '42110202':'means_000905.csv',
                '42120201':'means_sp.csv',
                '42120502':'means_hs.csv',
                '42400102':'means_gold.csv',
                #'42400400':'means_nh0100nhf.csv',
                '42400300':'means_spgscltrspi.csv',
            }
    for (viewid, ass) in assets.iteritems():
        print viewid, ass
        data_df = pd.read_csv(ass)
        create_dates = np.repeat(datetime.datetime.now(),len(data_df))
        update_dates = np.repeat(datetime.datetime.now(),len(data_df))
        union_tmp = {}
        union_tmp['ids'] = np.repeat(viewid, len(data_df))
        union_tmp['dates'] = data_df['date']
        union_tmp['create_time'] = create_dates
        union_tmp['update_time'] = update_dates
        union_tmp['means'] = data_df['means'] / 100.0
        union_tmp = pd.DataFrame(union_tmp)
        print insert_predict_pct(union_tmp)
