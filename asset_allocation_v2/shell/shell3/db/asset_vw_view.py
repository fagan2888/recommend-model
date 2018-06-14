#coding=utf8

from sqlalchemy import MetaData, Table, select, func
import pandas as pd
import logging
from . import database
from dateutil.parser import parse
logger = logging.getLogger(__name__)
def get_viewid_by_indexid(indexid):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('vw_view', metadata, autoload=True)
    columns = [
        t.c.globalid.label('viewid'),
    ]
    s = select(columns).where(t.c.vw_asset_id == indexid)
    df = pd.read_sql(s, db)
    return df

