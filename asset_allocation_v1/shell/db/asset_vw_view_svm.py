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

def get_newest_relative_view(viewid):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('vw_view_svm', metadata, autoload=True)
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
        strength = view_df['strength']
        probs = view_df['probs']
        create_at = view_df['create_time']
        update_at = view_df['update_time']
        d = {
                'vw_view_id':ids,
                'vw_date':dates,
                'vw_strength':strength,
                'vw_probility':probs,
                'created_at':create_at,
                'updated_at':update_at,
        }
        test_df = pd.DataFrame(data=d)
        result = (0, "Insert into vw_veiw_svm success")
        df = test_df.to_sql("vw_view_svm", db, if_exists='append', index=False)
    except Exception, e:
        result = (1, "Insert into vw_veiw_svm fail: " + e.message)
    return result
