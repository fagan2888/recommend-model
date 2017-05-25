#coding=utf8

from sqlalchemy import MetaData, Table, select, func
import pandas as pd
import logging
import database
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

def get_viewid_by_assids(ass1_id, ass2_id):
    """
    usage:得到viewid通过两个资产的id
    """
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('vw_view', metadata, autoload=True)
    columns = [
        t.c.globalid.label('viewid'),
    ]
    s = select(columns).where(t.c.vw_asset_id == ass1_id).\
        where(t.c.vw_asset2_id == ass2_id)
    df = pd.read_sql(s, db)
    return df

