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

#
# tc_timing
#
def load_assets(id_):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('ra_bl_asset', metadata, autoload=True)

    columns = [
        t1.c.bl_asset_id,
    ]

    s = select(columns)
    if id_ is not None:
        s = s.where(t1.c.globalid == id_)

    df = pd.read_sql(s, db)

    return df.bl_asset_id.values
